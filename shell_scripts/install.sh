#!/bin/bash

# Define variable name and value
VAR_NAME="C2S_HOME"
VAR_VALUE=$PWD

# Detect the shell config file
if [[ "$SHELL" == */bash ]]; then
    CONFIG_FILE="$HOME/.bashrc"
elif [[ "$SHELL" == */zsh ]]; then
    CONFIG_FILE="$HOME/.zshrc"
else
    echo "Unsupported shell: $SHELL"
    exit 1
fi

# Add export to shell config if not already present
if grep -q "export $VAR_NAME=" "$CONFIG_FILE"; then
    echo "Variable $VAR_NAME already defined in $CONFIG_FILE"
else
    echo "export $VAR_NAME=\"$VAR_VALUE\"" >> "$CONFIG_FILE"
    echo "Added $VAR_NAME to $CONFIG_FILE"
fi
