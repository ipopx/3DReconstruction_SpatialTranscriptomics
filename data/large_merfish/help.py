import scanpy as sc
adata = sc.read_h5ad("./animal4_int.h5ad")
cut_adata = adata[:5,:]
cut_adata.write_h5ad("./animal4_int_5cells.h5ad")