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

# Function to prompt for input if not provided
get_project_name() {
    while true; do
        if [ -z "$1" ]; then
            log_info "Enter a project name:"
            read PROJECT_NAME
        else
            PROJECT_NAME="$1"
        fi

        if gh repo view "jlawman/$PROJECT_NAME" &>/dev/null; then
            log_error "A repository named '$PROJECT_NAME' already exists."
            log_info "Please choose a different name."
            set -- "" # Clear the argument to force a prompt in the next iteration
        else
            break
        fi
    done
} 