# Skynet TB command
Skynet Side:
tensorboard --logdir ./logs --port 6001

Local:
W3:
ssh -N -f -L localhost:16006:localhost:6001 dyung6@143.215.131.34

Skynet:
ssh -N -f -L localhost:16006:localhost:6001 dyung6@sky1.cc.gatech.edu

# How to remote access Tensorboard running on server
https://gist.github.com/mrnabati/009c43b9b981ec2879008c7bff2fbb22

# tmux
# Reattach to session:
tmux attach -t <screen number>

# Kill session
ctrl+a+b -> : then type 'kill-session'

# Create new session with name
tmux new -s <name>

# Detach session
ctrl+a+b -> : then type 'detach'

# List tmux sessions
'tmux ls'

# Scroll in screen
`ctrl-a esc`

# Kill process
`sudo pkill -9 <PID>` in terminal
or
`sudo kill -9 <PID>` in terminal

# Kill all processes
`sudo killall -u dyung6`