tryCatch( {
	library(flowCore)
}, error = function(e){
	source("http://bioconductor.org/biocLite.R")
	biocLite("flowCore")
	library(flowCore)
})
tryCatch({
	library(magrittr)
}, error = function(e){
	install.packages('magrittr')
	library(magrittr)
})

	
destfile <- "Levine_32dim_notransform.fcs"
### Check if file exists and download if it does not
if (!file.exists(destfile)) {
	download.file("https://flowrepository.org/experiments/817/fcs_files/102553/download",
		 destfile=destfile) 
}

read.FCS(destfile, 
	transformation = FALSE, 
	truncate_max_range = FALSE) -> data

#########################
### ARCSINH TRANSFORM ###
#########################
scale <- 5
cols_to_scale <- 3:36

exprs(data) -> df_transform
exprs(data) -> df_notransform
asinh( df_transform[, cols_to_scale] / scale) -> df_transform[, cols_to_scale]


write.table(df_transform, file="Levine_32dimtransform.csv", quote=FALSE, sep=',', row.names = FALSE)
write.table(df_notransform, file="Levine_32dim_notransform.csv", quote=FALSE, sep=',', row.names = FALSE)

