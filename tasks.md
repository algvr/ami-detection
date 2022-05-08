# Analysis of Electrocardiograms via Artificial Neural Networks For a Reliable Assessment of a Possible Myocardial Infarction
### Tools

- PyCharm for working with Python: https://www.jetbrains.com/pycharm/download/#section=windows
  - use community edition; students can also get Pro edition for free
- Git: https://git-scm.com/download/win
  - Git cheat sheet: https://wac-cdn.atlassian.com/dam/jcr:e7e22f25-bba2-4ef1-a197-53f46b6df4a5/SWTM-2088_Atlassian-Git-Cheatsheet.pdf?cdnVersion=255
- Git LFS for datasets: https://git-lfs.github.com/

### Organization

- make task list & keep updated
  - take notes for problems & thoughts regarding each task
- make GitHub repo
  - first, private
- agree on programming language
  - Python, unless anyone prefers another one
- tools/libraries (depends on programming language; assume Python)
  - pillow (Python Imaging Library) for synthesizing/augmenting ECG images: https://pillow.readthedocs.io/en/stable/
  - PyTorch for working with neural networks
  - torchvision / torchvision.transforms for data augmentations
    - try doing data augmentation using pillow first!
    - check out https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py for some examples
    - more comprehensive documentation: https://pytorch.org/vision/stable/transforms.html
    - may need to convert NumPy arrays into PyTorch Tensors
- split tasks
  - keep tasks fine-grained for better workload balancing; not too much to avoid overhead
- if you have coding questions, don't hesitate to ask :)

### Building new dataset

- note: working with dataset will require approx. 3-4 GBs of RAM
  - change code to only load part of DS
- datasets
  - start with PTB-XL to set up pipeline
  - can expand later to improve
    - https://www.frontiersin.org/articles/10.3389/fcvm.2021.654515/full fuses multiple datasets
- reading in data from PTB-XL
  - check out "example_physionet.py"
- collect ECG templates to superimpose graphs on
  - can generate ourselves using imaging libraries, or use a free template (e.g. from https://www.printablepaper.net/preview/ECG_Paper)
- superimpose ECG lines on ECG template
  - start with single template, then expand
  - initial calibration mark: check "Calibration" under https://thoracickey.com/ecg-interpretation/
  - take care of paper speed
  - take care to map leads correctly to parts of ECG paper
    - _important_: we should do research whether ECGs always have the layout, or there can be variations
      - standard layout: https://en.wikipedia.org/wiki/Electrocardiography#Lead_locations_on_an_ECG_report
    - need to ask user the layout of the ECG/number of leads in the web tool to be sure
  - there are different types of ECG (different number of leads)
    - standard and maximum is 12-lead (also lead type of PTB-XL); you can extract individual leads from that
- stochastic data augmentation for images
  - shearing, rotation, blurring, noise, _shadows_: determine sensitive parameters (degrees of rotation, standard deviation of noise/blur, ...) by trial and error
    - rotation/noise/blur cannot be very big; we assume the user takes some care in taking the photo: something like https://www.frontiersin.org/files/Articles/654515/fcvm-08-654515-HTML-r2/image_m/fcvm-08-654515-g002.jpg, but maybe a bit more rotated/shadowed/noisy, not like https://previews.123rf.com/images/belchonock/belchonock1502/belchonock150207815/36929822-paper-sheet-on-table-close-up.jpg
  - same ECG time series can result in different images due to randomness of data augmentation
    - can obtain large dataset
    - need to label new images according to ID of time series used, to avoid data leakage from training set into test set for a fair evaluation (e.g. for single time series TS0001: TS0001_01, TS0001_02, etc.)
  - scaling and adding background images
    - ideas for background images
      - different kinds of tables (color, illumination)
      - different kinds of walls
      - uniform color (something like https://unsplash.com/photos/gM8igOIP5MA; then use image editor to get same background in blue, red, white, ...)
      - can take table/wall background images ourselves
        - then use image editor to get same background in blue, red, white, ...
      - alternatively, look for public domain images to avoid licensing issues
        - https://www.publicdomainpictures.net/en/
        - https://unsplash.com/
  - images can have very different resolution; try to match that of common cameras
    - images will be big, but we will downsize them before feeding them into neural network
- physical ECG imitations
  - printing ECGs
    - only do after verified that all else taken care of
    - determine amount of ECGs to print
  - taking photos of ECGs
    - use some variety of cameras/phones
  - can take image of same ECG imitation from different angles, using different cameras/illumination conditions/backgrounds, etc.
    - again, need to label photos according to ECG imitation used to avoid data leakage from training into test set

### Classification

- preprocessing before feeding into neural network
  - downsizing
    - adaptive resizing, determined from "red cell" size on ECG paper?
  - increasing contrast
  - rectification
    - are there libraries for this?
    - alternatively, can add markers on paper for easier rectification (or even use red cells from ECG paper)
- neural networks for classification
  - we need to be able to deal with _thin_ lines; possible networks:
    - U-Net (add more skip connections?), U-Net++
    - DenseNet
    - GL-Dense-U-Net
  - split dataset into train/test/validation set & perform hyperparameter tuning
  - perform evaluation
  - compare networks
  - train different networks for 1-lead ECGs, etc.?
- if time left/image-based neural networks have poor performance
  - try image to time series transcription algorithm
  - possible networks to use for time series:
    - RNN
    - LSTM
    - GRU
  - again: train different networks for 1-lead ECGs, etc.?
  - can compare performance to image-based neural networks
  - maybe try different classes/networks, report how well which classes work

### Presentation

- determine structure, e.g.
  - explain MI problematic & our idea
  - explain our approaches
  - explain challenges we faced
  - evaluation
  - web tool demo?
  - future work
- evaluation: add diagrams/tables from evaluation of neural networks
- web tool demo: add screenshots, maybe even short video
- future work: create app for even quicker scans, ...

### Web tool

- for backend, reuse pipeline from https://algvrithm.com/face-generator/
- add questionnaire form for symptoms to tackle NSTEMI issue/obtain more reliable diagnosis?
  - different symptoms for men and women; do research
- add different translations?
  - build in a way that allows for easy translation
- need to ask user the layout of the ECG/number of leads in the web tool to be sure
- add some info about MIs on web tool page
- add disclaimers

## Questions and answers from the consultation hour for the project

- From covid project: beware of parameters? most probably, beware of model overfitting with the neural network and how we label the dataset. Justify our understanding of the model's behaviours and interpretation of results.
- For our project: Justify conceptual system by meta-analysis studies. ECGs legal implications, how to ensure privacy concerns in the webtool apps. For the disclaimer: check ETH spinoffs for the ones checking medical data. Check legal ecosystem via the spinoffs but not important at the moment. Ask Vaiva for asking the legal board of ETH. Recheck error types (false positives and negatives), compare with false positives/negatives in medicine. Maybe some image features provoke errors of a certain type. Model error propagation and (importantly) characterize different error types across models. Check how other health tools treat the asymmetry between type I error and type II error. Idea: visualize attention layer, then cross-check with expert (e.g. cardiologist) whether highlighted features are indeed indicative of an MI / a normal ECG. Important: Create ROC curve. Make different types of errors intuitively understandable for users on the website. Compare image (CNN)-based and time series-based ECG classification
- From Cyber-sec: Make easy-to-understand problem formulation. Why this problem, and what insights are obtained.
- From Weather-Traffic: Check how bias/variance can affect the model's prediction

## Some to-dos we discussed during 02 May lecture
 - Generate data with less leads (other format is common ECG prints), maybe after most stuff is done, it reqs. changing just some lines of code
 - Write a disclaimer about the use of the webtool app
 - Write down instructions and general text for the webapp
 - 