################################################################################
################ Solver Filter
######### Paul-Marie Grollemund 
######### 2022-05-13 
################################################################################
#### Clean up ----
rm(list=ls())

#### Répertoire de travail ----
setwd("~/Documents/MCF/Recherche/PRIVABIO/code/developpement_empreinte/Filter_solver/")

#### Packages nécessaires ----
library(bmp)
library(magick)
library(pracma)
library(FactoMineR)
library(factoextra)

#### Fonctions nécessaires ---- 
hamming_distance <- function(u,v) sum(u != v)
norm_hamming_distance <- function(u,v) sum(u != v) / length(v)
least_square <- function(image1,image2) sum((image1-image2)^2)
norm_least_square <- function(image1,image2){
 image1 <- image1 /255
 image2 <- image2 /255
 mean((image1-image2)^2)
}
score <- function(image_test,image_ref,
                  template_test,template_ref,
                  w=0.5){
 w * norm_hamming_distance(template_test,template_ref) + 
  (1-w) * norm_least_square(image_test,image_ref)
}

image_preprocess <- function(magick_image,size=p){
 format <- paste(size,"x",size,sep="")
 image_scale(magick_image,format)
}
Sobel_convolution <- function(magick_image,matrix_filter){
 res_convolution <- image_convolve(magick_image,matrix_filter)
 res_convolution <- image_data(res_convolution,channels = "gray")[1,,]
 res_convolution <- apply(res_convolution,2,as.integer)
 
 res_convolution
}
random_transformation <- function(m,random_GS_matrix=transformation_matrix){
 m %*% random_GS_matrix
}
discretization <- function(m){
 as.integer(colSums(m) > 0)
}
process <- function(magick_image,M=10,filter_matrix=H_Sobel){
 #### Sobel convolution ---- 
 res_filter <- Sobel_convolution(magick_image,H_Sobel)
 
 #### Random matrix transformation ----
 res_transformation <- random_transformation(res_filter)
 
 #### Discretization ---- 
 template <- discretization(res_transformation)
 
 return(template)
}

#### Options ---- 
path_image <- "~/Documents/MCF/Recherche/PRIVABIO/data/Data_Basis_Finger_Print/FVC2004DB4/bmp/"
H_Sobel <- matrix(c(1, 2, 1, 0, 0, 0, -1, -2, -1), nrow = 3)
M <- 30
p <- 100

#### Random matrix (transformation) ---- 
set.seed(5)
gaussian_matrix <- matrix(rnorm(p*M,0,1),
                          nrow=p,ncol=M)
transformation_matrix <- gramSchmidt(gaussian_matrix)$Q

#### Target image ---- 
image <- image_read(file.path(path_image,"I1_S1.bmp"))
image <- image_preprocess(image)

matrix_target <- image_data(image,channels = "gray")[1,,]
matrix_target <- apply(matrix_target,2,as.integer)
template_target <- process(image)

#### Attack image ---- 
image_attack <- image_read(file.path(path_image,"I1_S3.bmp"))
image_attack <- image_preprocess(image_attack)

matrix_attack <- image_data(image_attack,channels = "gray")[1,,]
matrix_attack <- apply(matrix_attack,2,as.integer)
template_attack <- process(image_attack)

score(matrix_target, matrix_attack, 
      template_target, template_attack)

cbind(template_target,
      template_attack)
sum(template_target != template_attack)
hamming_distance(template_target,template_attack)

#### Importer les données ---- 
files <- list.files(path_image)
image_tmp <- image_read(
  file.path(path_image,files[1])
)
image_tmp <- image_preprocess(image_tmp)

matrix_target <- image_data(image_tmp,channels = "gray")[1,,]
dims <- dim(matrix_target)

images <- matrix(NA,ncol=prod(dims),nrow=length(files))
for(i in 1:length(files)){
  image_tmp <- image_read(
    file.path(path_image,files[i])
    )
  image_tmp <- image_preprocess(image_tmp)
  
  matrix_target <- image_data(image_tmp,channels = "gray")[1,,]
  matrix_target <- apply(matrix_target,2,as.integer)
  images[i,] <- as.vector(matrix_target)
}
# images <- rbind(
#   as.vector(matrix_attack),
#   images
# )

#### Appliquer ACP ---- 
res_acp <- PCA(images,graph = FALSE,ncp=min(length(files),20))
res_acp_save <- res_acp
fviz_screeplot(res_acp)

starting_image <- apply(res_acp_save$ind$coord,2,mean)
dim_range <- apply(res_acp_save$ind$coord,2,range)
dim_sd <- apply(res_acp_save$ind$coord,2,sd)
weight <- res_acp$eig[,2]
N <- min(length(weight),20)
weight <- weight[1:N]

####  X0 ---- 
n <- nrow(matrix_target)
p <- ncol(matrix_target)
# matrix_X0_pca <- res_acp_save$ind$coord[15,]
matrix_X0_pca <- res_acp_save$ind$coord[1,]
res_acp$ind$coord <- t(matrix_X0_pca)
matrix_X0 <- FactoMineR::reconst(res_acp)
matrix_X0 <- matrix(matrix_X0,nrow=dims[1],ncol=dims[2])
# matrix_X0 <- matrix(sample(0:255,n*p,replace = T),nrow=n,ncol=p)
# matrix_X0 <- matrix_attack
magick_X0 <- magick::image_read(as.raster(t(matrix_X0)/255))
magick_X00 <- magick_X0
template0 <- process(magick_X0)

score0 <- score(matrix_X0, matrix_attack,
      template_target, template0)
cbind(template_target,template0)
sum(template_target != template0)

n_loop <- 10
n_iter <- rep(c(500,100),n_loop/2)
N_value <- 10
res_score <- c(score0,rep(NA,sum(n_iter)))
res_score_immuable <- c(score0,rep(NA,sum(n_iter)))
ws <- rep(c(0.7,0.3),n_loop/2)
current_iter = 0
for(iter_loop in 1:n_loop){
  for(iter in 1:n_iter[iter_loop]){
    current_iter = current_iter + 1
    i <- sample(1:N,1,prob = weight)
    
    
    values <- runif(N_value,dim_range[1,i],dim_range[2,i])
    res_score_tmp <- rep(NA,N_value)
    res_score_immuable_tmp <- rep(NA,N_value)
    for(k in 1:N_value){
      matrix_X0_pca_tmp <- matrix_X0_pca
      matrix_X0_pca_tmp[i] <- values[k]
      
      res_acp$ind$coord <- t(matrix_X0_pca_tmp)
      matrix_X0_tmp <- FactoMineR::reconst(res_acp)
      matrix_X0_tmp <- matrix(matrix_X0_tmp,nrow=dims[1],ncol=dims[2])
      matrix_X0_tmp[ matrix_X0_tmp<0 ] <- 0
      
      magick_X0_tmp <- magick::image_read(as.raster(t(matrix_X0_tmp)/255))
      template0_tmp <- process(magick_X0_tmp)
      
      res_score_tmp[k] <- score(matrix_X0_tmp, matrix_attack, template_target, template0_tmp,ws[iter_loop])
      res_score_immuable_tmp[k] <- score(matrix_X0_tmp, matrix_attack, template_target, template0_tmp)
    }
    
    index <- which.min(res_score_tmp)[1]
    
    if(res_score_tmp[index] <= res_score[current_iter]){
      matrix_X0_pca_tmp <- matrix_X0_pca
      matrix_X0_pca_tmp[i] <- values[index]
      matrix_X0_pca <- matrix_X0_pca_tmp
      
      res_acp$ind$coord <- t(matrix_X0_pca)
      matrix_X0 <- FactoMineR::reconst(res_acp)
      matrix_X0 <- matrix(matrix_X0,nrow=dims[1],ncol=dims[2])
      matrix_X0_tmp[ matrix_X0_tmp<0 ] <- 0
      
      res_score[current_iter+1] <- res_score_tmp[index]
      res_score_immuable[current_iter+1] <- res_score_immuable_tmp[index]
    }else{
      res_score[current_iter+1] <- res_score[current_iter]
      res_score_immuable[current_iter+1] <- res_score_immuable[current_iter]
    }
  }
}


plot(res_score,type="l")
lines(res_score_immuable,col="red")

magick_X0 <- magick::image_read(as.raster(t(matrix_X0)/255))
template0 <- process(magick_X0)

cbind(template_target,template_attack,template0)
sum(template_target != template_attack)
sum(template_target != template0)

par(mfrow=c(2,2))
plot(image_attack)
plot(magick_X0)
plot(image)
plot(magick_X00)
par(mfrow=c(1,1))
least_square(matrix_X0,matrix_attack)



plot(res_acp_save$ind$coord[,1:2],pch=16)
points(matrix_X0_pca[1],matrix_X0_pca[2],col="red",pch=16)
