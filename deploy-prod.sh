#!/bin/bash

python3 -m venv senti-venv

source senti-venv/bin/activate
python3 -m pip install -q gdown

gdown --folder --continue --folder "https://drive.google.com/drive/folders/1S0qKIyL5-Yu6XXLSeEkLlQfEJrU7WBGE" -O ./Models

EXCLUDE_FILE="exclude-sync.txt"

if [ $# -ne 4 ]; then
    echo "Usage: $0 <remote_user> <remote_host> <ssh_port> <ssh-key>"
    echo "Example: $0 username 192.168.1.100 22 ~/.ssh/id-rsa"
    exit 1
fi

REMOTE_USER=$1
REMOTE_HOST=$2
SSH_PORT=$3
SSH_KEY=$4

SOURCE="./"
REMOTE_PATH="~/projects/senti"
ABSOLUTE_REMOTE_PATH="/home/"$REMOTE_USER"/projects/senti"

SERVICE_NAME="senti"
START_COMMAND="$ABSOLUTE_REMOTE_PATH/venv/bin/streamlit run Scripts/strmlt.py --server.port 8502"

ssh -i $SSH_KEY -p $SSH_PORT "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH" && \
rsync -avzP --delete -e "ssh -o \"StrictHostKeyChecking no\" -i $SSH_KEY -p $SSH_PORT" --exclude-from="$EXCLUDE_FILE" "$SOURCE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "Transfer completed successfully"
else
    echo "Transfer failed"
fi

ssh -i $SSH_KEY -p $SSH_PORT "$REMOTE_USER@$REMOTE_HOST" bash -s << EOF
# Setup virtual environment and install dependencies
cd $REMOTE_PATH
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

# Create systemd service file
sudo bash -c "cat > /etc/systemd/system/${SERVICE_NAME}.service" << 'ENDSERVICE'
[Unit]
Description=Sentiment analysis
After=network.target

[Service]
Type=simple
User=${REMOTE_USER}
WorkingDirectory=${ABSOLUTE_REMOTE_PATH}
Environment="PATH=${ABSOLUTE_REMOTE_PATH}/venv/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=${START_COMMAND}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
ENDSERVICE

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service
sudo systemctl start ${SERVICE_NAME}.service
sudo systemctl status ${SERVICE_NAME}.service
EOF

echo "Deployment and service setup completed"
exit 1
