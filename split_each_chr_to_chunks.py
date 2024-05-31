import sys


def split_X_to_chunks(gene_end_file_name,input_path, gene_output_end_file_name, chunk_size,chr,rep):
def split_X_to_chunks(input_file, output_path, chunk_size):
    from_ = 0
    to_ = chunk_size
    print("in")
    print(chr)
    gene_output_file = '/sise/nadav-group/nadavrap-group/hadasa/my_storage/impoving_PRS/data/our_model/splited_bed_files_for_all_chro/rep'+rep+'/' + str(chr) + gene_output_end_file_name
    with open(gene_output_file, 'wb') as gene_file_handle:
        file_name = input_path +'chr' + str(chr) + gene_end_file_name

    with open(output_file, 'wb') as gene_file_handle:
        file_name = input_file
        print(file_name)
        G = read_plink1_bin(file_name, verbose=False)  # xarray.core.dataarray.DataArray
        while True:  # while there are still samples
@@ -36,20 +34,14 @@ def split_X_to_chunks(gene_end_file_name,input_path, gene_output_end_file_name,


def main():
    chr = sys.argv[1]
    rep = sys.argv[2]
    print (chr)
    chunk_size=750
    #train
    split_X_to_chunks(gene_end_file_name="_X_train_no_cov_no_missing.bed",input_path='/sise/nadav-group/nadavrap-group/hadasa/my_storage/impoving_PRS/data/training_bed_files/rep'+rep+"/",
                     gene_output_end_file_name="_X_train_1k_chunks_no_missing.pkl",
                     chunk_size=chunk_size,chr=chr,rep=rep)
    #test
    split_X_to_chunks(gene_end_file_name="_X_test_no_cov_no_missing.bed",input_path='/sise/nadav-group/nadavrap-group/hadasa/my_storage/impoving_PRS/data/test_bed_files/rep'+rep+'/',
                     gene_output_end_file_name="_X_test_1k_chunks_no_missing.pkl",
                     chunk_size=chunk_size,chr=chr,rep=rep)

    file_to_split = sys.argv[1]
    output_file = sys.argv[2]
    chunk_size = 1000

    split_X_to_chunks(input_file=file_to_split,
                     output_file=output_file,
                     chunk_size=chunk_size)
]
if __name__ == "__main__":
    main()
