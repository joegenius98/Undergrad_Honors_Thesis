#!/bin/bash


if [ $# -lt 2 ]; then
    echo "Usage: $0 <port> <log_dir>"
    exit 1
fi

port=$1
log_dir=${2:-""}

[ -d "$log_dir" ] && echo "Directory "$log_dir" exists."

# Check if the log directory exists
if [ ! -z "$log_dir" ] && [ ! -f "$log_dir" ]; then
  echo "Error: Log directory $log_dir does not exist"
  exit 1
fi

# Start the visdom server
nohup python -m visdom.server -port $port &> visdom_server.out &

# If a log directory is provided, replay the log using Visdom
if [ ! -z "$log_dir" ] ; then
  nohup python -c "import visdom; vis = visdom.Visdom(port=$port); vis.replay_log('$log_dir')" &> vis_replay_script.out &
fi


