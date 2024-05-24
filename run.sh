echo "Running run_bert.sh"
sh run_bert.sh

echo "Running run_rnn.sh"
sh run_rnn.sh

echo "Running metrics.py"
python3 metric.py