#!/bin/bash

# Source utility scripts
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils/logging.sh"
source "$SCRIPT_DIR/utils/browser.sh"
source "$SCRIPT_DIR/utils/doppler-utils.sh"

# Initialize skip flags
SKIP_GITHUB=false
SKIP_VERCEL=false
SKIP_WEBSITES=false

# Parse command line arguments
for arg in "$@"; do
  if [[ "$arg" == "--skip" ]]; then
    SKIP_GITHUB=true
    SKIP_VERCEL=true
    SKIP_WEBSITES=true
    log_info "Skipping GitHub, Vercel setup, and website opening as requested"
  elif [[ "$arg" == "--skip-github" ]]; then
    SKIP_GITHUB=true
    log_info "Skipping GitHub setup as requested"
  elif [[ "$arg" == "--skip-vercel" ]]; then
    SKIP_VERCEL=true
    log_info "Skipping Vercel setup as requested"
  elif [[ "$arg" == "--skip-websites" ]]; then
    SKIP_WEBSITES=true
    log_info "Skipping website opening as requested"
  fi
done

# Function to prompt for input if not provided
get_project_name() {
    while true; do
        if [ -z "$1" ] || [[ "$1" == --* ]]; then
            log_info "Enter a project name:"
            read PROJECT_NAME
        else
            PROJECT_NAME="$1"
        fi

        # Only check for existing repo if we're not skipping GitHub
        if [ "$SKIP_GITHUB" = false ] && gh repo view "jlawman/$PROJECT_NAME" &>/dev/null; then
            log_error "A repository named '$PROJECT_NAME' already exists."
            log_info "Please choose a different name."
            set -- "" # Clear the argument to force a prompt in the next iteration
        else
            break
        fi
    done
}

# Function to open URL in default browser
open_url() {
    if command -v open &> /dev/null; then
        open "$1"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$1"
    elif command -v start &> /dev/null; then
        start "$1"
    else
        log_warning "Unable to open $1 automatically. Please visit it manually."
    fi
}

# Get project name from argument or prompt
# Filter out any --skip arguments
ARGS=()
for arg in "$@"; do
    if [[ ! "$arg" == --* ]]; then
        ARGS+=("$arg")
    fi
done
get_project_name "${ARGS[0]}"

# Set variables
BASE_DIR="$HOME/Documents/adder/100-concepts"
FULL_PATH="$BASE_DIR"/$PROJECT_NAME
TEMPLATE="jlawman/red-panda-simple"

log_section "🚀 CREATING PROJECT: $PROJECT_NAME"

# Create the directory
log_step "Creating project directory"
mkdir -p "$FULL_PATH"

# Change to the new directory
cd "$BASE_DIR" || exit

# Create the GitHub repository if not skipped
if [ "$SKIP_GITHUB" = false ]; then
    log_section "GITHUB SETUP"
    log_step "Creating GitHub repository from template"
    gh repo create "$PROJECT_NAME" --template "$TEMPLATE" --private --clone
else
    log_info "Skipping GitHub repository creation"
    # If we're skipping GitHub, we need to manually clone the template
    log_step "Cloning template repository"
    git clone "https://github.com/$TEMPLATE.git" "$PROJECT_NAME"
    # Remove the .git directory to disconnect from the template repo
    rm -rf "$PROJECT_NAME/.git"
    # Initialize a new git repository
    cd "$PROJECT_NAME" || exit
    git init
    git add .
    git commit -m "Initial commit from template"
    cd "$BASE_DIR" || exit
fi

# Only show Vercel setup instructions and perform Vercel setup if not skipped
if [ "$SKIP_VERCEL" = false ]; then
    # Display beautiful instructions for Vercel setup
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo -e "║                    ${CYAN}🚀 VERCEL SETUP GUIDE 🚀${RESET}                    ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                                                                ║"
    echo -e "║  When prompted for vercel settings:                            ║"
    echo "║                                                                ║"
    echo -e "║  ${GREEN}1. Use the default settings for most options${RESET}                  ║"
    echo -e "║  ${RED}2. IMPORTANT: When asked about the directory to deploy,       ║"
    echo -e "║     specify 'frontend' instead of the default${RESET}                ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    cd "$FULL_PATH" || exit
    log_section "VERCEL SETUP"

    # Link to Vercel project directly specifying frontend as the root directory
    log_step "Linking to Vercel project with frontend as root directory"
    vercel link --project "$PROJECT_NAME"
    vercel git connect
else
    log_info "Skipping Vercel setup"
    cd "$FULL_PATH" || exit
fi

# Set up Doppler project and populate with secrets from template
log_section "DOPPLER SETUP"
if command -v doppler &> /dev/null; then
    setup_doppler "$PROJECT_NAME" "$FULL_PATH" "red-panda-simple"
else
    log_error "Doppler CLI not found. Please install Doppler CLI to set up secrets management."
    log_info "Visit https://docs.doppler.com/docs/install-cli for installation instructions."
fi

# Ask if user wants to track analytics
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo -e "║                    ${CYAN}📊 ANALYTICS SETUP 📊${RESET}                       ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo -e "║  Would you like to set up Fathom Analytics for this project?   ║"
echo -e "║  This will create a Fathom site and add the site ID to Doppler ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
read -p "$(echo -e "${YELLOW}Set up analytics? (Y/n):${RESET} ")" setup_analytics
setup_analytics=${setup_analytics:-Y}  # Default to Y if empty

# Set up Fathom Analytics site and add to Doppler if user wants to
if [[ "$setup_analytics" =~ ^[Yy]$ ]]; then
    log_section "FATHOM ANALYTICS SETUP"
    
    # Get the directory where the current script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    # Try multiple possible locations for the fathom-setup.sh file
    FATHOM_SCRIPT=""
    
    # Check in the same directory as this script
    if [ -f "$SCRIPT_DIR/fathom-setup.sh" ]; then
        FATHOM_SCRIPT="$SCRIPT_DIR/fathom-setup.sh"
    # Check in the setup directory relative to the current directory
    elif [ -f "./setup/fathom-setup.sh" ]; then
        FATHOM_SCRIPT="./setup/fathom-setup.sh"
    # Check in the parent directory's setup folder
    elif [ -f "../setup/fathom-setup.sh" ]; then
        FATHOM_SCRIPT="../setup/fathom-setup.sh"
    fi
    
    if [ -n "$FATHOM_SCRIPT" ]; then
        # Check if FATHOM_API_TOKEN is set
        if [ -z "$FATHOM_API_TOKEN" ]; then
            log_warning "FATHOM_API_TOKEN environment variable is not set."
            
            # Try to get the token from Doppler
            if command -v doppler &> /dev/null; then
                log_info "Attempting to get Fathom API token from Doppler..."
                FATHOM_API_TOKEN=$(doppler secrets get FATHOM_API_KEY --plain 2>/dev/null || echo "")
                
                if [ -n "$FATHOM_API_TOKEN" ]; then
                    log_success "Successfully retrieved Fathom API token from Doppler."
                else
                    log_info "Could not retrieve Fathom API token from Doppler."
                    log_info "Skipping Fathom Analytics setup."
                fi
            else
                log_info "Doppler CLI not found. Cannot retrieve Fathom API token."
                log_info "Skipping Fathom Analytics setup."
            fi
        fi
        
        # Only proceed if we have a token
        if [ -n "$FATHOM_API_TOKEN" ]; then
            # Source the Fathom setup script to use its functions
            log_step "Using Fathom setup script: $FATHOM_SCRIPT"
            source "$FATHOM_SCRIPT"
            
            # Create Fathom site
            log_step "Creating Fathom site"
            
            # Check if create-fathom-site.sh exists in the same directory as fathom-setup.sh
            CREATE_SCRIPT_DIR=$(dirname "$FATHOM_SCRIPT")
            CREATE_SCRIPT="$CREATE_SCRIPT_DIR/create-fathom-site.sh"
            
            if [ ! -f "$CREATE_SCRIPT" ]; then
                log_error "Failed to find create-fathom-site.sh script at $CREATE_SCRIPT"
                log_info "This script is required to create a Fathom Analytics site."
                log_info "Please ensure create-fathom-site.sh is in the same directory as fathom-setup.sh."
                continue
            fi
            
            # Check if the script is executable
            if [ ! -x "$CREATE_SCRIPT" ]; then
                log_warning "create-fathom-site.sh is not executable. Attempting to make it executable..."
                chmod +x "$CREATE_SCRIPT"
                if [ ! -x "$CREATE_SCRIPT" ]; then
                    log_error "Failed to make create-fathom-site.sh executable."
                    log_info "Please run: chmod +x $CREATE_SCRIPT"
                    continue
                fi
            fi
            
            # Run the create_fathom_site function with debug output
            log_info "Executing create_fathom_site function with project name: $PROJECT_NAME"
            site_id_output=$(create_fathom_site "$PROJECT_NAME" 2>&1)
            create_result=$?
            
            if [ $create_result -eq 0 ]; then
                # Extract the site ID from the last line of output
                SITE_ID=$(echo "$site_id_output" | tail -n 1)
                
                # Add site ID to Doppler
                log_step "Adding site ID to Doppler"
                add_to_doppler "$PROJECT_NAME" "$SITE_ID" "$FATHOM_API_TOKEN"
                
                log_success "Fathom Analytics setup complete."
            else
                log_error "Failed to create Fathom Analytics site."
                log_info "Error details:"
                echo "$site_id_output" | sed 's/^/    /'
                log_info "To debug, try running these commands manually:"
                log_info "    cd $FULL_PATH"
                log_info "    $FATHOM_SCRIPT $PROJECT_NAME"
            fi
        fi
    else
        log_error "Fathom setup script not found. Please ensure fathom-setup.sh is in the setup directory."
        log_info "Looked in: $SCRIPT_DIR, ./setup/, and ../setup/"
    fi
else
    log_info "Skipping Fathom Analytics setup as requested."
fi

# Install npm dependencies in the app folder
log_section "DEPENDENCIES INSTALLATION"
log_step "Installing npm dependencies"
cd "$FULL_PATH/frontend" || exit
npm i
cd "$FULL_PATH" || exit

#vercel.json starting point - but broke things
# {
#     "builds": [
#       {
#         "src": "app/**/*",
#         "use": "@vercel/next"
#       }
#     ]
#   }

log_section "OPENING PROJECT"
# Open the repository in GitHub Desktop
if command -v github &> /dev/null; then
    github "$FULL_PATH"
    log_success "Repository opened in GitHub Desktop."
else
    log_warning "GitHub Desktop command not found. Please ensure GitHub Desktop is installed and the CLI tool is in your PATH."
    log_info "You can open the project manually in GitHub Desktop."
fi

# Check if Cursor is installed and in PATH
if command -v cursor &> /dev/null; then
    cursor "$FULL_PATH"
    log_success "Project $PROJECT_NAME has been created and opened in Cursor."
else
    log_warning "Cursor command not found. Please ensure Cursor is installed and in your PATH."
    log_info "You can open the project manually at: $FULL_PATH"
    # Attempt to open the directory in Finder
    open "$FULL_PATH"
fi

# Only open websites if not skipped
if [ "$SKIP_WEBSITES" = false ]; then
    log_section "OPENING WEBSITES"
    log_step "Opening required websites"
    open_websites
else
    log_info "Skipping website opening as requested."
fi

log_section "✨ PROJECT SETUP COMPLETE ✨"
echo -e "${GREEN}${BOLD}"
echo "  _____ _                 _____    ____        _ _     _ "
echo " |_   _(_)_ __ ___   ___|_   _|__| __ ) _   _(_) | __| |"
echo "   | | | | '_ \` _ \ / _ \ | |/ _ \  _ \| | | | | |/ _\` |"
echo "   | | | | | | | | |  __/ | | (_) | |_) | |_| | | | (_| |"
echo "   |_| |_|_| |_| |_|\___| |_|\___/|____/ \__,_|_|_|\__,_|"
echo "                                                           "
echo -e "${RESET}"
echo -e "${CYAN}${BOLD}Project Name:${RESET} ${PROJECT_NAME}"
echo -e "${CYAN}${BOLD}Location:${RESET} ${FULL_PATH}"
echo -e "${CYAN}${BOLD}Template:${RESET} ${TEMPLATE}"
echo ""