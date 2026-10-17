#!/usr/bin/env bash
# ==============================================================================
# TarjimaAI macOS Double-Clickable Desktop Launcher
# Double-click this file in Finder to launch TarjimaAI and open your browser!
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

chmod +x ./run.sh
./run.sh
