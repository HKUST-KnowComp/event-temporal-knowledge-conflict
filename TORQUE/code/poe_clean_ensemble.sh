for seed in 0 41 42
do
for file in poe poe_mixin poe_mixin_h1 poe_mixin_h5e-1 poe_mixin_h1e-1 poe_mixin_h5e-2 poe_mixin_h1e-2
do
# file_path=/home/data/zwanggy/2023/temp_rel_bias/${file}_baseline
file_path=/home/data/zwanggy/2023/temp_rel_bias/matres/${file}_baseline_seed${seed}
python poe_clean_ensemble.py --clean_path ${file_path} --file_name model.pt
done
done

# poe_mixin_h5e-2 poe_mixin_h1e-2