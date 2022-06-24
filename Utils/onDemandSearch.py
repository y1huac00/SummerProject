import commonTools
import customizedYaml
import os
import pandas as pd
import shutil


def move_to_error_bucket(source, dest):
    """
    Copy the image file under certain folder into a destination
    source: source path should be the EXACT path of the original file
    dest: destination file path could be a path, will be created if not exist.
    """
    if not os.path.exists(dest):
        os.mkdir(dest)
    if not os.path.exists(source):
        return
    shutil.copy2(source, dest)


def check_file_existance(source_path, annotation_path, dest_path):
    candidate_mags = ['40X', '80X', '100X']
    if os.path.isfile(source_path):
        return source_path, annotation_path, dest_path
    for cand in candidate_mags:
        new_source = source_path.replace('50X', cand)
        new_dest = dest_path.replace('50X', cand)
        new_annotation = annotation_path.replace('50X', cand)
        if os.path.isfile(new_source):
            return new_source, new_annotation, new_dest
    return source_path, annotation_path, dest_path


def find_genus(record, genus):
    matches = record.loc[record[8] == genus]
    matched_grids = matches[[0, 1, 2]]
    matched_grids[2] = matched_grids[2].apply(str)
    matched_grids['folder_path'] = matched_grids[0] + '_' + matched_grids[1] + '_50X'
    matched_grids['grid_path'] = matched_grids['folder_path'] + '_grid_' + matched_grids[2] + '.tif'
    matched_grids['image_path'] = matched_grids['folder_path'] + '.tif'
    matched_grids['annotation_path'] = matched_grids['folder_path'] + '_grid_' + matched_grids[2] + '.xml'
    result_length = len(matched_grids)
    print(f'Find {result_length} matched results of {genus} in the dataset')
    return matched_grids


def process_grids(matched_grids, grid_path, output_path, image_dir ,annotation_path):
    """
    the annotation_path parameter left for adjusting if new file format applied.
    """
    for idx, row in matched_grids.iterrows():
        grid_dest_path = os.path.join(output_path, row['folder_path'])
        grid_source_folder_path = os.path.join(grid_path, row['folder_path'])
        annotation_source_folder_path = os.path.join(annotation_path, row['folder_path'])
        grid_source_path = os.path.join(grid_source_folder_path, row['grid_path'])
        annotation_source_path = os.path.join(annotation_source_folder_path, row['annotation_path'])
        grid_source_path, annotation_source_path, grid_dest_path = \
            check_file_existance(grid_source_path, annotation_source_path, grid_dest_path)
        if os.path.isfile(grid_source_path):
            # Reuse the function
            move_to_error_bucket(grid_source_path, grid_dest_path)
            move_to_error_bucket(annotation_source_path, grid_dest_path)

    all_images = matched_grids['image_path'].value_counts()
    # for val, cnt in all_images.iteritems():
    #     image_source_path = os.path.join(image_dir, val)
    #     image_source_path, _, _ = test_file_existance(image_source_path,'',output_path)
    #     if os.path.isfile(image_source_path):
    #         move_to_error_bucket(image_source_path, output_path)

if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_path = yaml_data.get_data('base_path')
    out_path = yaml_data.build_new_path('base_path', 'outputs')
    grid_path = yaml_data.build_new_path('base_path', 'grid_images')
    image_path = yaml_data.build_new_path('base_path', 'raw_images')
    yaml_data.data['genus_annotation'] = yaml_data.build_new_path('base_path', 'genus_annotation')
    annotation_path = yaml_data.build_new_path('genus_annotation', 'pascal_voc')
    record_file = yaml_data.build_new_path('base_path', 'all_records.csv')
    all_records = pd.read_csv(record_file, header=None)

    # This part coould be replaced by if condition to switch the search target
    genus_keywaord = 'pistocythereis'
    matched_grids = find_genus(all_records, genus_keywaord)
    process_grids(matched_grids,grid_path,out_path,image_path,annotation_path)
