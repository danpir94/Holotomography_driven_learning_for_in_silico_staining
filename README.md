# Holotomography-driven learning for in-silico staining

This repository accompanies the paper *[Pirone, D., Giugliano, G., Schiavo, M., Montella, A., Mugnano, M., Cerbone, V., Raia, M., Scalia, G., Kurelac, I., Medina, D. L., Capasso, M., Iolascon, A., Memmolo, P. & Ferraro, P. (2025). Holotomography-driven learning for in-silico staining of single cells in flow cytometry avoiding co-registration. bioRxiv, 2025-07. https://doi.org/10.1101/2025.07.22.666145]*.  
It provides the MATLAB® R2024b code and dataset to reproduce and extend the results of our proposed Holotomography-driven deep learning framework for the in-silico staining of single-cell Quantitative Phase Maps (QPMs) acquired under flow cytometry conditions.

---

## Dataset

The `Dataset` folder includes:  
- Training set  
- Validation set  
- Internal test set  
- Independent internal test set  
- Independent external test set  

Each sample consists of an experimental 2D QPM and its corresponding nucleus mask, obtained using the Computational Segmentation based on Statistical Inference (CSSI) algorithm applied to Holo-Tomographic Flow Cytometry data.

---

## Pretrained Model & Results

To reproduce the results shown in the paper:
- Use the pretrained network 'trainedNet_paper.mat'
- Use the corresponding performance metrics saved in 'metrics_trainedNet_paper.mat'

To display input, target, and predicted images across the dataset:
1. Open 'main_metrics.m';
2. Set 
   - name = 'trainedNet_paper';
   - isShow = 1;
3. Run the script.

To plot F1 score and IoU metrics:
1. Open 'main_boxplot.m'
2. Set:
   name = 'trainedNet_paper';
3. Run the script.

---

## Retrain and Re-evaluate the Model

If you wish to retrain the Holotomography-driven CNN (e.g., after modifying the architecture or training parameters):
1. Run 'main_training.m'.
The resulting trained network will be saved automatically as 'trainedNet-YYYY-MM-DD-HH-MM-SS-Epoch-N.mat', where the timestamp and number of epochs reflect the training session.

To evaluate a retrained model:
1. Open 'main_metrics.m';
2. Set:
   name = 'trainedNet-YYYY-MM-DD-HH-MM-SS-Epoch-N';
   isShow = 0;  % or 1 if you want to display images
3. Run the script.

To visualize updated boxplots:
1. Open 'main_boxplot.m';
2. Set:
   name = 'trainedNet-YYYY-MM-DD-HH-MM-SS-Epoch-N';
3. Run the script.

---

## License

Please cite the corresponding paper when using this repository.

---

## Contact

For questions, please contact:

- **Dr. Daniele Pirone**  
- daniele.pirone@isasi.cnr.it
