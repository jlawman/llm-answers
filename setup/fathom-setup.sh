#!/bin/bash

# Fathom Analytics Site Creation Script
# This script creates a new Fathom Analytics site and adds the site ID to Doppler environment variables

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to prompt for input if not provided
get_project_name() {
  if [ -z "$1" ]; then
    # Instead of prompting, use a default name or exit
    echo "Error: Project name is required. Usage: ./fathom-setup.sh <project-name>"
    exit 1
  else
    PROJECT_NAME="$1"
  fi
  echo "$PROJECT_NAME"
}

# Function to get Fathom API token
get_fathom_token() {
  # First try to get token from environment variable
  if [ -z "$FATHOM_API_TOKEN" ]; then
    # Try to get token from Doppler
    if command_exists doppler; then
      echo "🔍 Attempting to get Fathom API token from Doppler..."
      DOPPLER_TOKEN=$(doppler secrets get FATHOM_API_KEY --plain 2>/dev/null)
      
      if [ -n "$DOPPLER_TOKEN" ]; then
        # Truncate token for display
        token_length=${#DOPPLER_TOKEN}
        truncated_token="${DOPPLER_TOKEN:0:3}...${DOPPLER_TOKEN: -3}"
        echo "✅ Successfully retrieved Fathom API token from Doppler (${truncated_token})"
        echo "$DOPPLER_TOKEN"
        return 0
      fi
    fi
    
    # If we get here, we couldn't get the token from Doppler either
    echo "⚠️  FATHOM_API_TOKEN environment variable is not set."
    echo "Please set it before running this script:"
    echo "export FATHOM_API_TOKEN=your_token_here"
    
    # Ask if the user wants to enter the token now
    read -p "🔑 Do you want to enter your Fathom API token now? (y/n): " enter_token
    
    if [[ "$enter_token" == "y" || "$enter_token" == "Y" ]]; then
      read -p "🔒 Enter your Fathom API token: " input_token
      if [ -n "$input_token" ]; then
        export FATHOM_API_TOKEN="$input_token"
        # Truncate token for display
        token_length=${#input_token}
        truncated_token="${input_token:0:3}...${input_token: -3}"
        echo "✅ Token set: ${truncated_token}"
        echo "$FATHOM_API_TOKEN"
        return 0
      else
        echo "❌ No token provided. Exiting."
        exit 1
      fi
    else
      echo "👋 Exiting. Please set FATHOM_API_TOKEN and try again."
      exit 1
    fi
  else
    # Truncate existing token for display
    token_length=${#FATHOM_API_TOKEN}
    truncated_token="${FATHOM_API_TOKEN:0:3}...${FATHOM_API_TOKEN: -3}"
    echo "✅ Using existing Fathom API token: ${truncated_token}"
  fi
  echo "$FATHOM_API_TOKEN"
}

# Function to add Fathom site ID to Doppler
add_to_doppler() {
  local project_name="$1"
  local site_id="$2"
  local api_token="$3"
  
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📦 ADDING FATHOM SITE ID TO DOPPLER"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔹 Project: $project_name"
  echo "🔹 Site ID: $site_id (production/staging)"
  echo "🔹 Dev Site ID: 00000"
  echo ""
  
  if command_exists doppler; then
    # Make sure we're only passing the actual values, not any log messages
    # Strip any whitespace or newlines
    site_id=$(echo "$site_id" | tr -d '[:space:]')
    
    echo "⚙️  Setting site ID in dev environment..."
    doppler secrets set NEXT_PUBLIC_FATHOM_SITE_ID="00000" --project "$project_name" --config dev --silent
    echo "   ✅ Dev environment updated with ID: 00000"
    
    echo "⚙️  Setting site ID in staging environment..."
    doppler secrets set NEXT_PUBLIC_FATHOM_SITE_ID="$site_id" --project "$project_name" --config stg --silent
    echo "   ✅ Staging environment updated with ID: $site_id"
    
    echo "⚙️  Setting site ID in prod environment..."
    doppler secrets set NEXT_PUBLIC_FATHOM_SITE_ID="$site_id" --project "$project_name" --config prd --silent
    echo "   ✅ Production environment updated with ID: $site_id"
    
    echo ""
    echo "✅ SUCCESS: Fathom site ID added to all Doppler environments"
    echo "   NEXT_PUBLIC_FATHOM_SITE_ID (dev): 00000"
    echo "   NEXT_PUBLIC_FATHOM_SITE_ID (stg/prd): $site_id"
    echo ""
    return 0
  else
    echo ""
    echo "❌ ERROR: Doppler CLI not found. Please install Doppler CLI."
    echo "   Visit https://docs.doppler.com/docs/install-cli for installation instructions."
    echo ""
    return 1
  fi
}

# Function to create a Fathom site using the create-fathom-site.sh script
create_fathom_site() {
  local project_name="$1"
  
  # Format the project name: replace dashes with spaces and capitalize each word
  local formatted_name=$(echo "$project_name" | tr '-' ' ' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)} 1')
  
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🚀 CREATING FATHOM ANALYTICS SITE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔹 Site Name: \"$formatted_name\""
  echo "🔹 From Project: \"$project_name\""
  echo ""
  
  # Get the directory where this script is located
  local SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  local CREATE_SCRIPT="$SCRIPT_DIR/create-fathom-site.sh"
  
  # Call the create-fathom-site.sh script with the formatted name
  if [ -f "$CREATE_SCRIPT" ]; then
    echo "📄 Using create-fathom-site.sh script: $CREATE_SCRIPT"
    # Run the script and capture the raw output
    raw_site_id=$("$CREATE_SCRIPT" "$formatted_name")
    
    # Check if we got a valid site ID (should be alphanumeric)
    if [[ $raw_site_id =~ ^[A-Z0-9]+$ ]]; then
      echo ""
      echo "✅ SUCCESS: Fathom site created with ID: $raw_site_id"
      echo ""
      # Return just the site ID
      echo "$raw_site_id"
      return 0
    else
      echo ""
      echo "❌ ERROR: Failed to create Fathom site or invalid site ID returned:"
      echo "   $raw_site_id"
      echo ""
      return 1
    fi
  else
    echo ""
    echo "❌ ERROR: create-fathom-site.sh script not found at $CREATE_SCRIPT"
    echo "   Please ensure create-fathom-site.sh is in the same directory as this script."
    echo ""
    return 1
  fi
}

# Main function
main() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔍 FATHOM ANALYTICS SETUP"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  # Get project name
  PROJECT_NAME=$(get_project_name "$1")
  
  # Get Fathom API token
  FATHOM_API_TOKEN=$(get_fathom_token)
  
  # Create Fathom site using the create-fathom-site.sh script
  # and capture the site ID from the last line of output
  site_output=$(create_fathom_site "$PROJECT_NAME")
  create_result=$?
  
  if [ $create_result -eq 0 ]; then
    # Extract just the site ID from the last line of output
    SITE_ID=$(echo "$site_output" | tail -n 1)
    
    # Verify it's a valid site ID format
    if [[ $SITE_ID =~ ^[A-Z0-9]+$ ]]; then
      # Add site ID to Doppler
      add_to_doppler "$PROJECT_NAME" "$SITE_ID" "$FATHOM_API_TOKEN"
      
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "✨ SETUP COMPLETE!"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "🔹 Fathom site: \"$(echo "$PROJECT_NAME" | tr '-' ' ' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)} 1')\""
      echo "🔹 Site ID (stg/prd): $SITE_ID"
      echo "🔹 Site ID (dev): 00000"
      echo "🔹 Status: Site IDs added to all Doppler environments"
      echo ""
    else
      echo ""
      echo "❌ ERROR: Invalid site ID format: $SITE_ID"
      echo ""
      exit 1
    fi
  fi
}

# Check if script is being run directly or sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Script is being run directly
  main "$@"
fi 