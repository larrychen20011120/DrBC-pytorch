path=(model_result/100_200.pth model_result/200_300.pth model_result/1000_1200.pth model_result/2000_3000.pth model_result/4000_5000.pth)
for i in {0..29}
do

    graph="hw1_data/Synthetic/5000/${i}.txt"
    gt="hw1_data/Synthetic/5000/${i}_score.txt"
    python test.py --graph_path $graph --bc_path "$gt" --model_path "${path[0]}" --device gpu >> result.txt
    python test.py --graph_path $graph --bc_path "$gt" --model_path "${path[1]}" --device gpu >> result.txt
    python test.py --graph_path $graph --bc_path "$gt" --model_path "${path[2]}" --device gpu >> result.txt
    python test.py --graph_path $graph --bc_path "$gt" --model_path "${path[3]}" --device gpu >> result.txt
    python test.py --graph_path $graph --bc_path "$gt" --model_path "${path[4]}" --device gpu >> result.txt
done