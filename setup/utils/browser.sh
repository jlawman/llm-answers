#!/bin/bash

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

# Array of websites to open
websites=(
    "http://www.promptpromptprompt.com"
    "https://adder-analytics-ltd.sentry.io/projects/new/" #Switch to manual installation of token on general settings page
    #"https://app.usefathom.com/sites"
    "https://dashboard.clerk.com/apps/new"
    "https://clerk.com/docs/authentication/social-connections/google"
    #"https://console.cloud.google.com/welcome?project=redpanda-438423"
    "https://www.braintrust.dev/app/Adder"
    #"https://fal.ai/dashboard/keys"
    #"https://console.anthropic.com/settings/keys"
    #"https://console.groq.com/keys"
    #"https://platform.openai.com/api-keys"
    #"https://favicon.io/favicon-converter/"
    "https://crop-circle.imageonline.co/"
    "https://vercel.com/josh-lawmans-projects" #"https://vercel.com/new/josh-lawmans-projects"
    "https://ap.www.namecheap.com/"
    #"https://tailwindui.com/components"
    #"https://console.anthropic.com/dashboard"
    #"https://uploadthing.com/dashboard/new"
    #"https://app.pinecone.io/"
    #"https://signin.aws.amazon.com/signin?client_id=arn%3Aaws%3Asignin%3A%3A%3Aconsole%2Fcanvas&redirect_uri=https%3A%2F%2Fconsole.aws.amazon.com%2Fconsole%2Fhome%3FhashArgs%3D%2523%26isauthcode%3Dtrue%26nc2%3Dh_ct%26src%3Dheader-signin%26state%3DhashArgsFromTB_eu-north-1_8229a3adbe919477&page=resolve&code_challenge=mm8BX2OcMycsNlGtPPF0Vkvb1zOsa76NjIJQa0PVZco&code_challenge_method=SHA-256"
    #"https://dashboard.stripe.com/"
)

# Function to open all websites
open_websites() {
    for site in "${websites[@]}"; do
        open_url "$site"
        log_info "Opened: $site"
    done
} 