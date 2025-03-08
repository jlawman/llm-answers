#!/bin/bash

# Function to set up Doppler for a project
setup_doppler() {
    local PROJECT_NAME="$1"
    local FULL_PATH="$2"
    local TEMPLATE_PROJECT="$3"
    
    # Create a new Doppler project
    log_step "Creating Doppler project"
    doppler projects create "$PROJECT_NAME"
    
    # Clone secrets from template project using doppler-copy.sh
    log_info "Copying secrets from $TEMPLATE_PROJECT to $PROJECT_NAME..."
    
    # Get the directory where the current script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Check for doppler-copy.sh
    if [ -f "$PARENT_DIR/doppler-copy.sh" ]; then
        # Source the Doppler copy script to use its functions
        source "$PARENT_DIR/doppler-copy.sh"
        
        # Copy secrets from template project to new project
        log_step "Copying Doppler secrets from template"
        copy_doppler_secrets "$TEMPLATE_PROJECT" "$PROJECT_NAME"
        
        # Set up Doppler in the project directory
        cd "$FULL_PATH" || exit
        log_step "Configuring Doppler in project directory"
        doppler setup --project "$PROJECT_NAME" --config dev
        
        # Verify setup was successful
        log_info "Verifying Doppler setup..."
        SETUP_INFO=$(doppler configure)
        echo "$SETUP_INFO"
        
        log_success "Doppler project setup complete."
    else
        # Fallback to manual secret copying
        log_warning "Doppler copy script not found at $PARENT_DIR/doppler-copy.sh"
        log_info "Falling back to manual secret copying..."
        
        # Get all secrets from the template project
        log_info "Fetching all secrets from template project..."
        
        # Get a list of all secret names from the template project
        SECRET_NAMES=$(doppler secrets --project "$TEMPLATE_PROJECT" --config dev --only-names 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$SECRET_NAMES" ]; then
            # Copy each secret one by one
            echo "$SECRET_NAMES" | while read -r SECRET_NAME; do
                if [ -n "$SECRET_NAME" ]; then
                    # Get the secret value from the template project
                    SECRET_VALUE=$(doppler secrets get "$SECRET_NAME" --project "$TEMPLATE_PROJECT" --config dev --plain 2>/dev/null)
                    
                    # If the secret exists and has a value, set it in the new project
                    if [ -n "$SECRET_VALUE" ]; then
                        log_info "Setting $SECRET_NAME..."
                        doppler secrets set "$SECRET_NAME=$SECRET_VALUE" --project "$PROJECT_NAME" --config dev
                        doppler secrets set "$SECRET_NAME=$SECRET_VALUE" --project "$PROJECT_NAME" --config staging
                        doppler secrets set "$SECRET_NAME=$SECRET_VALUE" --project "$PROJECT_NAME" --config prod
                    fi
                fi
            done
            
            # Set up Doppler in the project directory
            cd "$FULL_PATH" || exit
            log_step "Configuring Doppler in project directory"
            doppler setup --project "$PROJECT_NAME" --config dev
            
            # Verify setup was successful
            log_info "Verifying Doppler setup..."
            SETUP_INFO=$(doppler configure)
            echo "$SETUP_INFO"
            
            log_success "Doppler project setup complete."
        else
            log_error "Failed to fetch secrets from template project. Please check if the template project exists and you have access to it."
        fi
    fi
}

# Function to test Doppler setup and copy functionality
test_doppler_setup() {
    local SOURCE_PROJECT="$1"
    local TARGET_PROJECT="$2"
    
    if [ -z "$SOURCE_PROJECT" ] || [ -z "$TARGET_PROJECT" ]; then
        log_error "Both source and target project names are required."
        log_info "Usage: test_doppler_setup <source_project> <target_project>"
        return 1
    fi
    
    log_section "DOPPLER TEST SETUP"
    log_info "Testing Doppler setup with source: $SOURCE_PROJECT and target: $TARGET_PROJECT"
    
    # Check if Doppler CLI is installed
    if ! command -v doppler &> /dev/null; then
        log_error "Doppler CLI not found. Please install Doppler CLI."
        log_info "Visit https://docs.doppler.com/docs/install-cli for installation instructions."
        return 1
    fi
    
    # Create the target project if it doesn't exist
    log_step "Checking if target project exists"
    if ! doppler projects | grep -q "$TARGET_PROJECT"; then
        log_info "Creating target project: $TARGET_PROJECT"
        doppler projects create "$TARGET_PROJECT"
        if [ $? -ne 0 ]; then
            log_error "Failed to create target project."
            return 1
        fi
        log_success "Target project created successfully."
    else
        log_info "Target project already exists."
    fi
    
    # Get the directory where the current script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Call the doppler-copy.sh script directly with proper arguments
    log_step "Copying secrets from $SOURCE_PROJECT to $TARGET_PROJECT"
    if [ -f "$PARENT_DIR/doppler-copy.sh" ]; then
        bash "$PARENT_DIR/doppler-copy.sh" "$SOURCE_PROJECT" "$TARGET_PROJECT"
        if [ $? -eq 0 ]; then
            log_success "Secrets copied successfully."
        else
            log_error "Failed to copy secrets."
            return 1
        fi
    else
        log_error "Doppler copy script not found at $PARENT_DIR/doppler-copy.sh"
        return 1
    fi
    
    log_success "Doppler test setup completed successfully."
    return 0
}

# Fix for the setup_doppler function to properly call doppler-copy.sh
fix_setup_doppler() {
    local PROJECT_NAME="$1"
    local TEMPLATE_PROJECT="$2"
    
    log_section "FIXING DOPPLER SETUP"
    log_info "Copying secrets from $TEMPLATE_PROJECT to $PROJECT_NAME"
    
    # Get the directory where the current script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Call the doppler-copy.sh script directly with proper arguments
    if [ -f "$PARENT_DIR/doppler-copy.sh" ]; then
        # Execute the script in a subshell to ensure it gets the arguments directly
        (cd "$PARENT_DIR" && ./doppler-copy.sh "$TEMPLATE_PROJECT" "$PROJECT_NAME")
        return $?
    else
        log_error "Doppler copy script not found at $PARENT_DIR/doppler-copy.sh"
        return 1
    fi
}

# Debug function to test doppler-copy.sh directly
debug_doppler_copy() {
    local SOURCE_PROJECT="$1"
    local TARGET_PROJECT="$2"
    
    if [ -z "$SOURCE_PROJECT" ] || [ -z "$TARGET_PROJECT" ]; then
        log_error "Both source and target project names are required."
        log_info "Usage: debug_doppler_copy <source_project> <target_project>"
        return 1
    fi
    
    log_section "DOPPLER COPY DEBUG"
    
    # Get the directory where the current script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Check if the script exists
    if [ ! -f "$PARENT_DIR/doppler-copy.sh" ]; then
        log_error "Script not found at $PARENT_DIR/doppler-copy.sh"
        return 1
    fi
    
    # Check if the script is executable
    if [ ! -x "$PARENT_DIR/doppler-copy.sh" ]; then
        log_info "Making script executable..."
        chmod +x "$PARENT_DIR/doppler-copy.sh"
    fi
    
    # Print the command that will be executed
    log_info "Executing: $PARENT_DIR/doppler-copy.sh \"$SOURCE_PROJECT\" \"$TARGET_PROJECT\""
    
    # Execute the script with arguments
    (cd "$PARENT_DIR" && ./doppler-copy.sh "$SOURCE_PROJECT" "$TARGET_PROJECT")
    
    # Check the return code
    local RESULT=$?
    if [ $RESULT -eq 0 ]; then
        log_success "Command executed successfully."
    else
        log_error "Command failed with exit code $RESULT."
    fi
    
    return $RESULT
} 