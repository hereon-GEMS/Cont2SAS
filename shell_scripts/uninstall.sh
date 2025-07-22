#!/bin/bash

# Variable name to remove
VAR_NAME="C2S_HOME"

# Detect the user's shell
if [[ "$SHELL" == */bash ]]; then
    CONFIG_FILE="$HOME/.bashrc"
elif [[ "$SHELL" == */zsh ]]; then
    CONFIG_FILE="$HOME/.zshrc"
else
    echo "Unsupported shell: $SHELL"
    exit 1
fi

# Backup the config file before editing
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

# Remove lines that export the variable
sed -i "/export $VAR_NAME=/d" "$CONFIG_FILE"

# Message
echo "Removed $VAR_NAME from $CONFIG_FILE"
