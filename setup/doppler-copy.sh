#!/bin/bash

# Define color codes for beautiful logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
RESET='\033[0m'

# Function for styled logging
log_info() {
    echo -e "${BLUE}ℹ️  ${BOLD}INFO:${RESET} $1"
}

log_success() {
    echo -e "${GREEN}✅ ${BOLD}SUCCESS:${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  ${BOLD}WARNING:${RESET} $1"
}

log_error() {
    echo -e "${RED}❌ ${BOLD}ERROR:${RESET} $1"
}

log_step() {
    echo -e "\n${CYAN}🔷 ${BOLD}STEP:${RESET} ${UNDERLINE}$1${RESET}"
}

log_section() {
    echo -e "\n${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${MAGENTA}${BOLD}  $1${RESET}"
    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

# Function to copy Doppler secrets from one project to another
copy_doppler_secrets() {
    local SOURCE_PROJECT="$1"
    local TARGET_PROJECT="$2"
    # Standard Doppler environment names
    local CONFIGS=("dev" "stg" "prd")
    local success_count=0
    local error_count=0
    
    log_section "DOPPLER SECRETS COPY"
    log_info "Copying secrets from $SOURCE_PROJECT to $TARGET_PROJECT..."
    
    # Check if Doppler CLI is installed
    if ! command -v doppler &> /dev/null; then
        log_error "Doppler CLI not found. Please install Doppler CLI."
        log_info "Visit https://docs.doppler.com/docs/install-cli for installation instructions."
        return 1
    fi
    
    # Check if source project exists
    log_info "Checking if source project '$SOURCE_PROJECT' exists..."
    if ! doppler projects | grep -q "$SOURCE_PROJECT"; then
        log_error "Source project '$SOURCE_PROJECT' does not exist."
        return 1
    else
        log_info "Source project '$SOURCE_PROJECT' found."
    fi
    
    # Check if target project exists
    log_info "Checking if target project '$TARGET_PROJECT' exists..."
    if ! doppler projects | grep -q "$TARGET_PROJECT"; then
        log_warning "Target project '$TARGET_PROJECT' does not exist."
        log_step "Creating target project: $TARGET_PROJECT"
        doppler projects create "$TARGET_PROJECT"
        if [ $? -eq 0 ]; then
            log_info "Target project '$TARGET_PROJECT' created successfully."
        else
            log_error "Failed to create target project '$TARGET_PROJECT'."
            return 1
        fi
    else
        log_info "Target project '$TARGET_PROJECT' found."
    fi
    
    # Process each config
    for CONFIG in "${CONFIGS[@]}"; do
        log_step "Processing $CONFIG environment"
        
        # Check if config exists in source project
        if ! doppler configs --project "$SOURCE_PROJECT" | grep -q "$CONFIG"; then
            log_warning "Config $CONFIG does not exist in source project. Skipping."
            continue
        fi
        
        # Check if config exists in target project, create if not
        if ! doppler configs --project "$TARGET_PROJECT" | grep -q "$CONFIG"; then
            log_info "Creating $CONFIG config in target project..."
            doppler configs create "$CONFIG" --project "$TARGET_PROJECT"
            if [ $? -ne 0 ]; then
                log_error "Failed to create $CONFIG config in target project."
                continue
            fi
        fi
        
        # Get all secrets from the source project for this config using download --no-file
        log_info "Fetching secrets from $CONFIG environment in source project..."
        
        # Create a temporary file to store secrets
        TEMP_FILE=$(mktemp)
        
        # Get all secrets in simple JSON format and directly set them in the target project
        doppler secrets download --project "$SOURCE_PROJECT" --config "$CONFIG" --no-file > "$TEMP_FILE" 2>/dev/null
        
        if [ $? -eq 0 ] && [ -s "$TEMP_FILE" ]; then
            log_info "Setting secrets in $CONFIG environment of target project..."
            
            # Import the JSON directly to the target project but suppress the output
            doppler secrets upload --project "$TARGET_PROJECT" --config "$CONFIG" "$TEMP_FILE" > /dev/null
            
            if [ $? -eq 0 ]; then
                # Get just the names of the secrets that were copied, excluding Doppler's internal variables
                SECRET_NAMES=$(jq -r 'keys[] | select(startswith("DOPPLER_") | not)' "$TEMP_FILE" | sort)
                SECRET_COUNT=$(echo "$SECRET_NAMES" | wc -l | tr -d ' ')
                
                echo -e "\n${CYAN}${BOLD}┌─────────────────────────────────────────────┐${RESET}"
                echo -e "${CYAN}${BOLD}│ ${RESET}${GREEN}✓${RESET} Copied ${BOLD}$SECRET_COUNT${RESET} secrets to ${BOLD}$CONFIG${RESET} environment ${CYAN}${BOLD}│${RESET}"
                echo -e "${CYAN}${BOLD}└─────────────────────────────────────────────┘${RESET}\n"
                
                # Only show the API keys and other important secrets (no categorization)
                echo -e "${CYAN}${BOLD}SECRETS:${RESET}"
                echo "$SECRET_NAMES" | while read -r SECRET_NAME; do
                    echo -e "  ${YELLOW}•${RESET} ${BOLD}$SECRET_NAME${RESET}"
                done
                
                echo ""  # Add a blank line for spacing
                log_success "Successfully copied secrets to $CONFIG environment."
                ((success_count++))
            else
                log_error "Failed to import secrets to $CONFIG environment."
                ((error_count++))
            fi
        else
            log_warning "No secrets found in $CONFIG environment of source project or failed to retrieve them."
        fi
        
        # Clean up
        rm "$TEMP_FILE"
    done
    
    log_success "Secrets copy operation completed."
    log_info "Summary: $success_count environments processed successfully, $error_count errors encountered."
    
    if [ "$error_count" -gt 0 ]; then
        log_warning "Some errors occurred during the copy process. Check the logs above for details."
        return 1
    else
        return 0
    fi
}

# Display help message
show_help() {
    echo -e "${CYAN}${BOLD}Doppler Secrets Copy Utility${RESET}"
    echo -e "A utility to copy secrets between Doppler projects"
    echo ""
    echo -e "${CYAN}${BOLD}Usage:${RESET}"
    echo -e "  ${BOLD}$0 <source_project> <target_project>${RESET}  Copy secrets from source to target"
    echo -e "  ${BOLD}$0 --help${RESET}                             Show this help message"
    echo ""
    echo -e "${CYAN}${BOLD}Examples:${RESET}"
    echo -e "  $0 my-source-project my-new-project"
}

# Parse command line arguments
parse_args() {
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_help
        exit 0
    elif [ $# -eq 2 ]; then
        copy_doppler_secrets "$1" "$2"
        exit $?
    else
        log_error "Invalid arguments. Expected 2 arguments: source_project and target_project."
        log_info "Got $# arguments: $@"
        show_help
        exit 1
    fi
}

# Main script execution - only run if this script is executed directly
# This prevents the parse_args from running when the script is sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
fi 