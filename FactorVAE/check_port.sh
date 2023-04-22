# Check if the visualization port is provided as an argument
if [ -z "$1" ]
then
  echo "Please provide the visualization port as an argument."
  exit 1
fi

# Check if the visualization port is an integer
if ! expr "$1" + 1 > /dev/null 2>&1
then
  echo "The visualization port must be an integer."
  exit 1
fi

# Check if visualization port is NOT being used by Visdom
process_ids=$(lsof -i :"$1" -t)
for pid in $process_ids; do
  if [ ! -z "$pid" ] && ps -p "$process_id" -o cmd | grep -q "visdom.server"
  then
      exit 0
  fi
done

echo "Please select a port being used for a Visdom server"
exit 1
