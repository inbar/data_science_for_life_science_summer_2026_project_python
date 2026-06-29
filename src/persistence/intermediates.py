def save_intermediate_dataset(dataset: MuData,
                              ):
    dataset_file_name = get_dataset_file_name()
    dataset_file_path = PROCESSED_DATA_DIR_PATH / dataset_file_name

    save_mudata_dataset_to_disk(dataset, dataset_file_path)