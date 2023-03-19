#prerequiste: pip install git+https://github.com/OhadRubin/halton
#NODE_ID=6 bash run.sh
export EXP_COUNT=v14debug
export WANDB_TAGS=$EXP_COUNT
export WANDB_NAME=$EXP_COUNT"_n$NODE_ID"

export HP_PATH=$(python -c "$(cat <<-EndOfMessage
from haltonpy.halton import grid_search,SearchSpace
space = SearchSpace("$WANDB_NAME")
spaces = [
    {"mv":[0.0,0.06,0.125,0.25,0.5],"lr":[5e-4,3e-4],"steps":[500001,500002,500003]},
    ]
groups = [list(range(1,31))]
# groups = [[1,2,3,4,5,6,7,8]]
for sp, group in zip(spaces,groups):
    space.add(sp,*group)
grid_search(**space.get($NODE_ID))
EndOfMessage
)")
echo $HP_PATH

source $(python -c "$(cat <<-EndOfMessage
from haltonpy.halton import set_environment_variables
config ="""
LR, lr, 5e-4
MV, mv, 0.0
STEPS, steps, 500000
---

"""
set_environment_variables(config, "$HP_PATH")
EndOfMessage
)")



echo "Starting experiment $WANDB_NAME"
# ███████╗████████╗ █████╗ ██████╗ ████████╗
# ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝
# ███████╗   ██║   ███████║██████╔╝   ██║   
# ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   
# ███████║   ██║   ██║  ██║██║  ██║   ██║   
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
python -c "import os;print(os.environ)" 

python src/train.py --config src/conf/linear_regression.yaml \
--training.grad_acum.min_value $MV \
--training.train_steps $STEPS \
--training.learning_rate  $LR \
--wandb.name $WANDB_NAME

# alerts msg --message "done! $WANDB_NAME"
