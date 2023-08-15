# Urban Region Embedding via Multi-View Contrastive Prediction

These are the source codes for the **ReCP** model and its corresponding data.

- Data
  1. dataset\attribute_m.pickle — region attribute matrix, each dimension corresponds to the number of POIs with a specific category in the region.
  2. dataset\source_matrix.pickle — outflow matrix, each dimension corresponds to the number of trips made by all individuals originating from the region.
  3. dataset\destina_matrix.pickle — inflow matrix, each dimension corresponds to the number of trips made by all individuals destined for the region.
  4. dataset\poi_info.pickle— region attribute data, including POI category ID, and POI category name.
  5. model_data\mh_cd.json — region land use data, the data format is [region id: land use type id].
  6. model_data\popus_count.npy — region popularity label.

- Code
  1. main.py — A file to run the ReCP model. 
  2. recp_data.py — A file to load data. 
  3. model.py — A file containing the details of model training and optimization.
  4. loss.py — A file containing the implementation details of all components of the model.
  5. configure.py — The hyper-parameters, the training options are defined in this file.
  
- Region Representations
  embeddings/ — These are the trained region representations for 10 times. 

- Tasks
  1. tasks.py — A file to calculate NMI, ARI, F-measure on the land usage clustering task, and MAE, RMSE, R^2 on the popularity prediction task.
