This repository provides implementations for **Triplet Discovery**, **Path Generation**, and **Adversary Attacks** using Knowledge Graph Embeddings (KGE) and related techniques.

The data after the attack is saved in the "BaselineData" folder.

The following is the process of reproducing the attack.

## 1. Triplet Discovery

### 1.1 Using Trained DistMult Model to Extract Rank-1 Triplets

The following steps demonstrate how to extract Rank-1 triplets using the trained DistMult model on the **WN18RR** dataset.

1. **Navigate to the `KGEModels` directory:**

    ```bash
    cd TripletDiscovery/KGEModels
    ```

2. **Preprocess the dataset (WN18RR):**

    ```bash
    python -u preprocess.py --dataset_name WN18RR
    ```

3. **Wrangle the Knowledge Graph (KG):**

    ```bash
    python -u wrangle_KG.py WN18RR
    ```

4. **Select target triplets using the DistMult model:**

    ```bash
    python -u select_targets.py --model distmult --data WN18RR
    ```

### 1.2 Using Trained DistMult Model for Rank-1 Triplets in CompGCN

1. **Navigate to the `CompGCN` directory:**

    ```bash
    cd ../../
    cd TripletDiscovery/CompGCN
    ```

2. **Select target triplets:**

    ```bash
    python select_target.py -name WN18RR -score_func conve -opn corr -data WN18RR -gpu 0
    ```

### 1.3 Merging Rank-1 Triplets Shared by DistMult and CompGCN

1. **Merge the extracted Rank-1 triplets:**

    ```bash
    cd ../../
    python merge.py
    ```

## 2. Path Generation

### 2.1 Install Dependencies for `torchdrug` Package

1. **Navigate to the `torchdrug` directory:**

    ```bash
    cd torchdrug
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    python setup.py install
    cd ..
    ```

### 2.2 Visualize Path Generation with AStarNet

1. **Navigate to the `PathGeneration` directory:**

    ```bash
    cd PathGeneration
    ```

2. **Run visualization script for the WN18RR dataset with AStarNet model:**

    ```bash
    python script/quick_visualize.py -c config/transductive/wn18rr_astarnet_visualize.yaml --checkpoint E:/PathAttack/PathGeneration/checkpoint/WN18RR_epoch_20.pth --gpus [0]
    ```
The generated path will be saved to the "Experiments" folder.

## 3. Adversary Attack

### 3.1 Triplet Deletion Attack

1. **Navigate to the appropriate directory:**

    ```bash
    cd ../
    ```

2. **Run the triplet deletion attack:**

    ```bash
    python select_targets.py --file_path PathGeneration/experiments/KnowledgeGraphCompletion/WN18RR/AStarNet/2025-11-25-20-21-25/WN18RR_del_distmult_compgcn_astar.txt
    ```
The output is the triplet to be deleted.

### 3.2 Triplet Addition Attack

1. **Navigate to the `KGEModels` directory:**

    ```bash
    cd TripletDiscovery/KGEModels
    ```

2. **Preprocess the dataset for triplet addition:**

    ```bash
    python -u preprocess.py --dataset_name WN18RR_add_original --add
    ```

3. **Wrangle the Knowledge Graph for triplet addition:**

    ```bash
    python -u wrangle_KG.py WN18RR_add
    ```

4. **Perform entity addition:**

    ```bash
    python -u dot_add_entity.py --data WN18RR_add
    ```

## Dependencies

- torchdrug (for path generation)

## References
Parts of this codebase are based on the code from following repositories:

- **AStarNet**: A method for Path Generation in Knowledge Graphs. [A*Net](https://github.com/DeepGraphLearning/AStarNet)
- **CompGCN**: https://github.com/malllabiisc/CompGCN
- **Attribution**: A method for Rank-1 Triplets Generation in Knowledge Graphs. https://github.com/PeruBhardwaj/AttributionAttack
- **InferenceAttack**: Test the performance under the poisoned dataset. https://github.com/PeruBhardwaj/InferenceAttack

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
