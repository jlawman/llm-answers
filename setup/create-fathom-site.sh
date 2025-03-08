#!/bin/bash

# Script to create a Fathom site and return the site ID
# Usage: ./create-fathom-site.sh "Site Name" [sharing] [share_password]

# Check if site name is provided
if [ -z "$1" ]; then
  echo "Error: Site name is required"
  echo "Usage: ./create-fathom-site.sh \"Site Name\" [sharing] [share_password]"
  exit 1
fi

# Get API token from Doppler
API_TOKEN=$(doppler secrets get FATHOM_API_KEY --plain)

if [ -z "$API_TOKEN" ]; then
  echo "Error: Failed to retrieve FATHOM_API_KEY from Doppler"
  exit 1
fi

# Set parameters
SITE_NAME="$1"
SHARING="${2:-none}"  # Default to 'none' if not provided
SHARE_PASSWORD="$3"   # Will be empty if not provided

# Build the JSON payload
if [ "$SHARING" = "private" ] && [ -n "$SHARE_PASSWORD" ]; then
  JSON_DATA="{\"name\":\"$SITE_NAME\",\"sharing\":\"$SHARING\",\"share_password\":\"$SHARE_PASSWORD\"}"
else
  JSON_DATA="{\"name\":\"$SITE_NAME\",\"sharing\":\"$SHARING\"}"
fi

# Make the API request
RESPONSE=$(curl -s -X POST https://api.usefathom.com/v1/sites \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$JSON_DATA")

# Check if the request was successful
if echo "$RESPONSE" | grep -q "\"id\""; then
  # Extract and output just the site ID
  SITE_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
  echo "$SITE_ID"
else
  # Output the error
  echo "Error creating site: $RESPONSE"
  exit 1
fi


#SITE_ID=$(./create-fathom-site.sh "My Website Name")
#echo "Created site with ID: $SITE_ID"