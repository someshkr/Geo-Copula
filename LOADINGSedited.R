library(readxl)

ALIPUR <- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Alipur-jan_19-aug_20.xlsx")
# ALIPUR 2019 PM2.5 DATA
S1<-as.data.frame(ALIPUR[1:365,],) [,5]

ANANDVIHAR<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/AnandVihar-jan_19-aug_20.xlsx")
S2<-as.data.frame(ANANDVIHAR[1:365,],) [,7]

ASKV<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/AshokVihar-jan_19-aug_20.xlsx")
S3<-as.data.frame(ASKV[1:365,],) [,7]

Ayanagar<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Ayanagar-jan_19-aug_20.xlsx")
S4<-as.data.frame(Ayanagar[1:365,],) [,12]

Bawana<-readxl:: read_excel("D:/Data Set of Delhi Air Pollution/Bawana-jan_19-aug_20.xlsx")
S5<-as.data.frame(Bawana[1:365,],) [,7]

BXing<-readxl:: read_excel("D:/Data Set of Delhi Air Pollution/Burari Crossing-jan_19-aug_20.xlsx")
S6<-as.data.frame(BXing[1:365,],) [,11]

CRRI<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/CRRI mathura road-jan_19-aug_20.xlsx")
S7<-as.data.frame(CRRI[1:365,],) [,12]

DTU<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/DTU-jan_19-aug_20.xlsx")
S8<-as.data.frame(DTU[1:365,],)[,19]

DS8<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Dwarka-Sector 8, Delhi - DPCC -jan_19-aug_20.xlsx")
S9<-as.data.frame(DS8[1:365,],) [,7]

# EAN<-readxl::read_excel("C:/Users/Lenovo/Desktop/SP/East Arjun Nagar, Delhi - CPCB -jan_19-aug_20.xlsx")
# S10<-as.data.frame(EAN[1:365,],) [,15]

IGI<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/IGI Airport (T3), Delhi - IMD -jan_19-aug_20.xlsx")
S10<-as.data.frame(IGI[1:365,],) [,11]

IHBAS<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/IHBAS, Dilshad Garden, Delhi - CPCB-jan_19-aug_20.xlsx")
S11<-as.data.frame(IHBAS[1:365,],) [,15]

ITO<- read_excel("D:/Data Set of Delhi Air Pollution/ITO, Delhi - CPCB-jan_19-aug_20.xlsx")
S12<-as.data.frame(ITO[1:365,],) [,17]


JPURI<-readxl:: read_excel("D:/Data Set of Delhi Air Pollution/Jahangirpuri, Delhi - DPCC-jan_19-aug_20.xlsx")
S13<-as.data.frame(JPURI[1:365,],) [,6]

JNS<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Jawaharlal Nehru Stadium, Delhi - DPCC -jan_19-aug_20.xlsx")
S14<-as.data.frame(JNS[1:365,],) [,7]

KSSR<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Karni Singh Shooting range-jan_19-aug_20.xlsx")
S15<-as.data.frame(KSSR[1:365,],) [,12]

Lodhi<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Lodhi Road, Delhi - IMD-jan_19-aug_20.xlsx")
S16<-as.data.frame(Lodhi[1:365,],) [,11]

DCNS<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Major Dhyan Chand National Stadium, Delhi - DPCC-jan_19-aug_20.xlsx")
S17<-as.data.frame(DCNS[1:365,],) [,7]

MM<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Mandir Marg, Delhi - DPCC-jan_19-aug_20.xlsx")
MM_19_old<-as.data.frame(MM[1:365,],) [,15]
S18<-as.numeric(MM_19_old)

Mundka<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Mundka, Delhi - DPCC-jan_19-aug_20.xlsx")
S19<-as.data.frame(DCNS[1:365,],) [,7]

NG<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Najafgarh, Delhi - DPCC-jan_19-aug_20.xlsx")
S20<-as.data.frame(NG[1:365,],) [,7]

NARELA<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Narela, Delhi - DPCC-jan_19-aug_20.xlsx")
S21<-as.data.frame(NARELA[1:365,],) [,6]

NN<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Nehru Nagar, Delhi - DPCC-jan_19-aug_20.xlsx")
S22<-as.data.frame(NN[1:365,],) [,7]

North_Campus<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/North Campus, DU, Delhi - IMD-jan_19-aug_20.xlsx")
S23<-as.data.frame(North_Campus[1:365,],) [,12]

NSIT_D<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/NSIT Dwarka, Delhi - CPCB-jan_19-aug_20.xlsx")
S24<-as.data.frame(NSIT_D[1:365,],) [,15]

OP2<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Okhla Phase-2, Delhi - DPCC-jan_19-aug_20.xlsx")
S25<-as.data.frame(OP2[1:365,],) [,6]

PPG<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Patparganj, Delhi - DPCC-jan_19-aug_20.xlsx")
S26<-as.data.frame(PPG[1:365,],) [,7]


PUSA_DPCC<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Pusa, Delhi - DPCC-jan_19-aug_20.xlsx")
S27<-as.data.frame(PUSA_DPCC[1:365,],) [,7]

PUSA_IMD<-read_excel("D:/Data Set of Delhi Air Pollution/Pusa, Delhi - IMD-jan_19-aug_20.xlsx")
S28<-as.data.frame(PUSA_IMD[1:365,],) [,12]

RKP<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/R K Puram, Delhi - DPCC-jan_19-aug_20.xlsx")
S29<-as.data.frame(RKP[1:365,],) [,18]

ROHINI<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Rohini, Delhi - DPCC-jan_19-aug_20.xlsx")
S30<-as.data.frame(ROHINI[1:365,],)[,12] 

Shadipur<-readxl::read_excel("D:/Data Set of Delhi Air Pollution/Shadipur, Delhi - CPCB-jan_19-aug_20.xlsx")
S31<-as.data.frame(Shadipur[1:365,],)[,15] 


SF<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Sirifort, Delhi - CPCB-jan_19-aug_20.xlsx")
S32<-as.data.frame(SF[1:365,],) [,19]
S32<-as.numeric(S32)

Sonia_Vihar<- read_excel("D:/Data Set of Delhi Air Pollution/Sonia Vihar, Delhi - DPCC-jan_19-aug_20.xlsx")
S33<-as.data.frame(Sonia_Vihar[1:365,],)[,7] 

Sri_AM<- read_excel("D:/Data Set of Delhi Air Pollution/Sri Aurobindo Marg^J Delhi - DPCC-jan_19-aug_20.xlsx")
S34<-as.data.frame(Sri_AM[1:365,],)[,7] 

Vivek_Vihar<- read_excel("D:/Data Set of Delhi Air Pollution/Vivek Vihar, Delhi - DPCC-jan_19-aug_20.xlsx")
S35<-as.data.frame(Vivek_Vihar[1:365,],)[,7] 

Wazirpur<- read_excel("D:/Data Set of Delhi Air Pollution/Wazirpur, Delhi - DPCC-jan_19-aug_20.xlsx")
S36<-as.data.frame(Vivek_Vihar[1:365,],)[,7] 

PB<- readxl::read_excel("D:/Data Set of Delhi Air Pollution/Punjabi Bagh, Delhi - DPCC-jan_19-aug_20.xlsx")
S37<-as.data.frame(PB[1:365,],) [,18]

# ---------------------------------------------------------------------------------
# ORIGINAL LOCATION FILE
LOCATION_ORG <- read.csv("D:/Data Set of Delhi Air Pollution/LOCATION.csv")


LOCATION.map<-data.frame(LOCATION=LOCATION_ORG[,1],EAST =LOCATION_ORG[,2],
                     NORTH =LOCATION_ORG[,3])


avl<-c(1:9,11:38)
# LOCATION FILE BASED ON AVAILABLE PM2.5 VALUES
LOCATION<-data.frame(LOCATION=LOCATION_ORG[,1][avl],EAST =LOCATION_ORG[,2][avl],
                     NORTH =LOCATION_ORG[,3][avl])
DF1<-data.frame(S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25,S26,S27,S28,S29,S30,S31,S32,S33,S34,S35,S36,S37)

dd1<-data.frame(DF1[305:334,1:37])

# AVERAGE PM2.5 FOR MONTHH OF NOVEMBER ACROSS 20 LOCATIONS
X_omega<-colMeans(dd1)
D1 <- data.frame(EAST =LOCATION_ORG[,2][avl],
                 NORTH =LOCATION_ORG[,3][avl],
                 X_omega=X_omega)


library(fitdistrplus)
nor_mle<-fitdistrplus::fitdist(X_omega,"norm")

u1<-pnorm(X_omega,mean=nor_mle$estimate['mean'],sd=nor_mle$estimate['sd'])
x2<-LOCATION_ORG[,2][avl]
x3<-LOCATION_ORG[,3][avl]
u2<-punif(x2, min =min(x2)-0.001, max = max(x2)+0.001)
u3<-punif(x3, min =min(x3)-0.001, max = max(x3)+0.001)
u<-matrix(c(u1,u2,u3),nrow = 37)
df <- cbind(X_omega,x2,x3)
df=as.data.frame(na.omit(df))
# 
# COPULA SELECTION 
library(copula)
cc1<-claytonCopula(0.010697,dim = 3)
# gofCopula(cc1,df,start=1)
param<-0.010697

mcc1<-mvdc(cc1,margins = c("norm","unif","unif"),paramMargins = list(list(mean=nor_mle$estimate['mean'],sd=nor_mle$estimate['sd']),list(min(x2)-0.001, max = max(x2)+0.001),list((x3)-0.001, max = max(x3)+0.001)))
# clayton.ml<-fitCopula(claytonCopula(param=0.061104,dim=3), u)

selectedcopula1<-VineCopula::BiCopSelect(u2,u3,familyset = NA)

# --------------------------------------------------------------------------------
library(sp)
library(geosphere)
library(dismo)
library(rgeos)
library(raster)
library(tidyverse)
library(sf)
setwd("D:/Data Set of Delhi Air Pollution")
# convert to sf format
dat_Delhi_sf <- getData(name = "GADM", 
                        country = "IND", 
                        level = 1)%>%
  # convert to simple features
  sf::st_as_sf() %>%
  # Filter down to NCT OF DELHI
  dplyr::filter(NAME_1 == "NCT of Delhi") 

# DF of only the coordinates
LOC<-as.data.frame(cbind(LOCATION$EAST,LOCATION$NORTH))
#LOC<-as.data.frame(cbind(LOCATION.map$EAST,LOCATION.map$NORTH))

xy <- SpatialPointsDataFrame(
matrix(c(LOCATION$EAST,LOCATION$NORTH), ncol=2), data.frame(ID=LOCATION$LOCATION),
 proj4string=crs(dat_Delhi_sf))


#xy <- SpatialPointsDataFrame(
 # matrix(c(LOCATION.map$EAST,LOCATION.map$NORTH), ncol=2), data.frame(ID=LOCATION.map$LOCATION),
  #proj4string=crs(dat_Delhi_sf))
# CREATION OF DISTANCE MATRIX
mdist <- distm(xy)
# HIERARCHICAL CLUSTERING
hc <- hclust(as.dist(mdist), method="complete")
tau<-matrix(0,nrow=37,ncol=37)
for(i in 1:ncol(dd1)){
  for(j in 1:ncol(dd1)){
    tau[i,j]<-VGAM::kendall.tau(dd1[,i],dd1[,j])
  }
}

colnames(tau) <- strtrim(LOCATION[,1],11)
rownames(tau) <- strtrim(LOCATION[,1],11)
heatmap(tau,Rowv = NA,Colv = NA,hclustfun = hclust(as.dist(mdist), method="complete"))
#install.packages("NbClust",dependencies = T)
library(factoextra)
library(NbClust)

#set.seed(123)
#fviz_nbclust(df, hcut, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")
library(ggplot2)
sort(hc$call)
par(mfrow=c(1,1))
A=fviz_nbclust(df, FUNcluster = hcut, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")
B=ggdendogram
model<-as.dendrogram(hc)
ddata <- dendro_data(model, type = "rectangle")
p <- ggplot(segment(ddata)) + 
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
  coord_flip() + 
  scale_y_reverse(expand = c(0.2, 0))
plot(hc,xlab="Distance")
abline(h=18026.508,col="red")

# define the distance threshold, in this case 18030 m =18.3km
#d=19800
d=19420
# define clusters based on a tree "height" cutoff "d" and add them to the SpDataFrame
xy$clust <- cutree(hc, h=d)
xy

# expand the extent of plotting frame
xy@bbox[] <- as.matrix(extend(extent(xy),0.01))

# get the centroid coords for each cluster
cent <- matrix(ncol=2, nrow=max(xy$clust))
for (i in 1:max(xy$clust))
  # gCentroid from the rgeos package
  cent[i,] <- gCentroid(subset(xy, clust == i))@coords

# compute circles around the centroid coords using a 18km radius
# from the dismo package
ci <- circles(cent, d=d, lonlat=T)

# plot
plot(ci@polygons, axes=T)
plot(xy, col=rainbow(max(xy$clust))[factor(xy$clust)], add=T)

# -------------------------CONSTRUCTION OF CIRCULAR RADIUS AROUND CLUSTERS---------------------------------------------------

centroid<-as.data.frame(cent)
dat_sf <- st_as_sf(centroid, coords = c("V1","V2"), crs = 4326)

# Buffer circles by d metres radius
dat_circles <- st_buffer(dat_sf, dist = d)
plot(Delhi_int_circles$CC_1)
plot(dat_circles,add=T)
plot(xy, col=rainbow(max(xy$clust))[factor(xy$clust)], add=T)

# Intersect the circles with the polygons
Delhi_int_circles <- st_intersection(dat_Delhi_sf, dat_circles)

bb <- st_bbox(Delhi_int_circles)
# xy1 is spatoal points of Easting & Northing only
xy1 <- SpatialPoints(matrix(c(LOCATION.map$EAST,LOCATION.map$NORTH), ncol=2),proj4string=crs(dat_Delhi_sf))


# ----------------EXTRACTING THE POINTS IN CIRCULAR POLYGON OF EACH CLUSTER-------------------------------------------------
library(secr)

# POLYGON OBJECTS FROM CLUSTERS
df1<-as.data.frame(Delhi_int_circles$geometry)[1,]
df2<-as.data.frame(Delhi_int_circles$geometry)[2,]
df3<-as.data.frame(Delhi_int_circles$geometry)[3,]
df4<-as.data.frame(Delhi_int_circles$geometry)[4,]

library(rgdal)
# CONSTRUCTION OF GRID IN ENTIRE DELHI NCR
data.shape<-readOGR(dsn="D:/Data Set of Delhi Air Pollution/Delhi_Boundary.shp")
crs(data.shape)<-crs(dat_Delhi_sf)
plot(data.shape)
grid <- makegrid(data.shape, cellsize = 0.004)
grid1 <- SpatialPoints(grid, proj4string =crs(data.shape),bbox =bbox(data.shape) )
plot(grid1,add=T)

# CONSTRUCTING THE GRID OF POINTS LYING IN EACH POLYGON CLUSTER
bool1<-pointsInPolygon(grid,df1 , logical = TRUE)
bool2<-pointsInPolygon(grid,df2 , logical = TRUE)
bool3<-pointsInPolygon(grid,df3 , logical = TRUE)
bool4<-pointsInPolygon(grid,df4 , logical = TRUE)

vec1<-as.vector(which(bool1, arr.ind=TRUE))
vec2<-as.vector(which(bool2, arr.ind=TRUE))
vec3<-as.vector(which(bool3, arr.ind=TRUE))
vec4<-as.vector(which(bool4, arr.ind=TRUE))

# POINTS OF CLUSTERS
cluster1_points<-grid[vec1,]
cluster2_points<-grid[vec2,]
cluster3_points<-grid[vec3,]
cluster4_points<-grid[vec4,]

# CREATING GRID OVERLAY ON SHAPEFILE
c1_sp<-SpatialPoints(cluster1_points, proj4string =crs(data.shape))
c2_sp<-SpatialPoints(cluster2_points, proj4string =crs(data.shape))
c3_sp<-SpatialPoints(cluster3_points, proj4string =crs(data.shape))
c4_sp<-SpatialPoints(cluster4_points, proj4string =crs(data.shape))   #CHECK


# OVERLAY CHECK

plot(data.shape)
plot(c1_sp)
plot(xy, col=rainbow(max(xy$clust))[factor(xy$clust)==1], add=T)
plot(c2_sp)
plot(xy, col=rainbow(max(xy$clust))[factor(xy$clust)==3], add=T)
plot(c3_sp)
plot(xy, col=rainbow(max(xy$clust))[factor(xy$clust)==1], add=T)
plot(c4_sp)
plot(xy, col=rainbow(max(xy$clust))[factor(xy$clust)==2], add=T)
# CHECK::::REGION OMITTED
# UNION ALL
union_1234<-union(union(union(cluster1_points,cluster2_points),cluster3_points),cluster4_points)
c1234_sp<-SpatialPoints(union_1234, proj4string =crs(dat_Delhi_sf))
# CHECKKKKKKKKKKK:REGION OMITTED
plot(data.shape)

plot(c1234_sp,add=T)

# -------------------------------------------------------------------------------------

# OBSERVED POINTS LYING IN CLUSTER 1
bool_1_obs<-pointsInPolygon(LOC,df1 , logical = TRUE)
vec1_obs<-as.vector(which(bool_1_obs, arr.ind=TRUE))
cluster1_obs_points<-LOC[vec1_obs,]
dim8<-nrow(cluster1_obs_points)

# OBSERVED POINTS LYING IN CLUSTER 2
bool_2_obs<-pointsInPolygon(LOC,df2 , logical = TRUE)
vec2_obs<-as.vector(which(bool_2_obs, arr.ind=TRUE))
cluster2_obs_points<-LOC[vec2_obs,]
dim4<-nrow(cluster2_obs_points)

# OBSERVED POINTS LYING IN CLUSTER 3
bool_3_obs<-pointsInPolygon(LOC,df3 , logical = TRUE)
vec3_obs<-as.vector(which(bool_3_obs, arr.ind=TRUE))
cluster3_obs_points<-LOC[vec3_obs,]
dim2<-nrow(cluster3_obs_points)

# OBSERVED POINTS LYING IN CLUSTER 4
bool_4_obs<-pointsInPolygon(LOC,df4 , logical = TRUE)
vec4_obs<-as.vector(which(bool_4_obs, arr.ind=TRUE))
cluster4_obs_points<-LOC[vec4_obs,]
dim1<-nrow(cluster4_obs_points)


plot(data.shape)
plot(Delhi_int_circles[,"CC_1"], main="Division of study area into some clusters")
plot(xy1,add=T,col=1:37,pch=7)
plot(1:2)
legend("left", legend = LOCATION[1:19,1], col = 1:19, pch =7, bty = "n")
legend("right", legend = LOCATION[20:37,1], col = 20:37, pch =7, bty = "n")
legend("top", legend = paste("LOC", 18:19), col = 18:19, pch =7, bty = "n")

# -------------------------------------------------------------------------
# presence_in_observed returns the presence vector 
# (i.e the region in which the point is lying among 15 disjoint sets forming 4 clusters)
# 0 denotes absence,1 denotes presence

presence_in_observed<-function(vv)
{
  bb1<-as.numeric(pointsInPolygon(vv,df1,logical = TRUE))
  bb2<-as.numeric(pointsInPolygon(vv,df2,logical = TRUE))
  bb3<-as.numeric(pointsInPolygon(vv,df3,logical = TRUE))
  bb4<-as.numeric(pointsInPolygon(vv,df4,logical = TRUE))
  presence_obs<-c(bb1,bb2,bb3,bb4)
  return(presence_obs)
}
l1<-presence_in_observed(LOC[1,])
l2<-presence_in_observed(LOC[2,])
l3<-presence_in_observed(LOC[3,])
l4<-presence_in_observed(LOC[4,])
l5<-presence_in_observed(LOC[5,])
l6<-presence_in_observed(LOC[6,])
l7<-presence_in_observed(LOC[7,])
l8<-presence_in_observed(LOC[8,])
l9<-presence_in_observed(LOC[9,])
l10<-presence_in_observed(LOC[10,])
l11<-presence_in_observed(LOC[11,])
l12<-presence_in_observed(LOC[12,])
l13<-presence_in_observed(LOC[13,])
l14<-presence_in_observed(LOC[14,])
l15<-presence_in_observed(LOC[15,])
l16<-presence_in_observed(LOC[16,])
l17<-presence_in_observed(LOC[17,])
l18<-presence_in_observed(LOC[18,])
l19<-presence_in_observed(LOC[19,])
l20<-presence_in_observed(LOC[20,])
l21<-presence_in_observed(LOC[21,])
l22<-presence_in_observed(LOC[22,])
l23<-presence_in_observed(LOC[23,])
l24<-presence_in_observed(LOC[24,])
l25<-presence_in_observed(LOC[25,])
l26<-presence_in_observed(LOC[26,])
l27<-presence_in_observed(LOC[27,])
l28<-presence_in_observed(LOC[28,])
l29<-presence_in_observed(LOC[29,])
l30<-presence_in_observed(LOC[30,])
l31<-presence_in_observed(LOC[31,])
l32<-presence_in_observed(LOC[32,])
l33<-presence_in_observed(LOC[33,])
l34<-presence_in_observed(LOC[34,])
l35<-presence_in_observed(LOC[35,])
l36<-presence_in_observed(LOC[36,])
l37<-presence_in_observed(LOC[37,])

df_presence_obs<-as.data.frame(rbind(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,
                                     l29,l30,l31,l32,l33,l34,l35,l36,l37))

#PRESENCE VECTOR OF ENTIRE DELHI
datalist=list()
for(i in 1:dim(union_1234)[1])
{
  datalist[[i]]<-presence_in_observed(union_1234[i,])
}
presence_delhi<-as.data.frame(do.call(rbind, datalist))

# -------------------------------------------------------------------------
find<-function(a,b,c,d)
{
  h1 <- mutate(presence_delhi, IDX = 1:n())
  h1<-filter(h1,V1==a & V2==b & V3==c & V4==d)
  h1 <- select(h1, IDX)[,1]
  r1<-SpatialPoints(union_1234[h1,], proj4string =crs(Delhi_int_circles))
  plot(data.shape)
  plot(Delhi_int_circles[,"CC_1"],add=T)
  plot(xy1,add=T,col=rep(1:20, each = 1),pch=20)
  plot(r1,add=T)
  return(r1)
}
# -------------------------------------------------------------
find_index<-function(a,b,c,d)
{
  h1 <- mutate(presence_delhi, IDX = 1:n())
  h1<-filter(h1,V1==a & V2==b & V3==c & V4==d)
  h1 <- select(h1, IDX)[,1]
  return(h1)
}
# ------------------------------------------------------------
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(9,19,20,24)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

# ----------------------------TEST DF--------------------------------------------
df_t_1<-find(0,0,0,1)
df_t_2<-find(0,0,1,0)
df_t_3<-find(0,0,1,1)
df_t_4<-find(0,1,0,0)
df_t_5<-find(0,1,0,1)
df_t_6<-find(0,1,1,0)
df_t_7<-find(0,1,1,1)
df_t_8<-find(1,0,0,0)
df_t_9<-find(1,0,0,1)
# df_t_10<-find(1,0,1,0)
# df_t_11<-find(1,0,1,1)
df_t_12<-find(1,1,0,0)
df_t_13<-find(1,1,0,1)
df_t_14<-find(1,1,1,0)
df_t_15<-find(1,1,1,1)

# DOESNT EXIST
# -----------------------------------------------------------------------------
# combination_presence denotes the 15 possible disjoint regions covering the 4 clusters
combination_presence<-rbind(c(0,0,0,1),c(0,0,1,0),c(0,0,1,1),c(0,1,0,0),c(0,1,0,1),c(0,1,1,0),c(0,1,1,1),
                            c(1,0,0,0),c(1,0,0,1),c(1,0,1,0),c(1,0,1,1),c(1,1,0,0),c(1,1,0,1),c(1,1,1,0),
                            c(1,1,1,1))
combination_presence<-as.data.frame(combination_presence)
vec<-seq(0,600,1)
# ------------------------------------------------------------------------
# testdf consists of all the points lying in regions 

# -----------------------FOR testdf1 (9,19,20,24)------------------------------------------------------
v1<-c(9,19,20,24)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(9,19,20,24)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))

DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------
                                           

# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(9,19,20,24)])
testdf1<-union_1234[find_index(0,0,0,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
FINAL<-interdf1
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 
# 2.367

# -------------------------------------------------------------------------------

# -----------------------FOR testdf2 (4,15,25,34)------------------------------------------------------
v1<-c(4,15,25,34)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(4,15,25,34)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(4,15,25,34)])
testdf1<-union_1234[find_index(0,0,1,0),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -------------------------------------------------------------------------------

# -----------------------FOR testdf3 (4,9,10,24)------------------------------------------------------
v1<-c(4,9,10,24)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(4,9,10,24)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))

DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(4,9,10,24)])
testdf1<-union_1234[find_index(0,0,1,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
FINAL<-rbind(FINAL,interdf1)
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# ------------------------------------------------------------------------

# -----------------------FOR testdf4 (11,26,33,35)------------------------------------------------------
v1<-c(11,26,33,35)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(11,26,33,35)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(11,26,33,35)])
testdf1<-union_1234[find_index(0,1,0,0),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -------------------------------------------------------------------------------
# -----------------------FOR testdf5 (9,19,20,24)------------------------------------------------------
v1<-c(19,20,24)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(9,19,20,24)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(19,20,24)])
testdf1<-union_1234[find_index(0,1,0,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -----------------------FOR testdf6 (2,7,12,13,14,15,16,17,25,35)------------------------------------------------------
v1<-c(2,7,12,13,14,15,16,17,25,35)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[v1],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))
LOC5<-rep(NA,length(vec))
LOC6<-rep(NA,length(vec))
LOC7<-rep(NA,length(vec))
LOC8<-rep(NA,length(vec))
LOC9<-rep(NA,length(vec))
LOC10<-rep(NA,length(vec))

DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4,LOC5,LOC6,LOC7,LOC8,LOC9,LOC10)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 5-------------------------------------------
lon<-LOC[v1[5],1]
lat<-LOC[v1[5],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,6]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[5,3]=parameter(vec,vec)[1]
PARAMETER1[5,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 6-------------------------------------------
lon<-LOC[v1[6],1]
lat<-LOC[v1[6],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,7]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[6,3]=parameter(vec,vec)[1]
PARAMETER1[6,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 7-------------------------------------------
lon<-LOC[v1[7],1]
lat<-LOC[v1[7],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[7]],na.rm=T),sd=sqrt(var(dd1[,v1[7]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[7]],na.rm=T),sd=sqrt(var(dd1[,v1[7]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[7]],na.rm=T),sd=sqrt(var(dd1[,v1[7]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,8]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[7,3]=parameter(vec,vec)[1]
PARAMETER1[7,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 8-------------------------------------------
lon<-LOC[v1[8],1]
lat<-LOC[v1[8],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[8]],na.rm=T),sd=sqrt(var(dd1[,v1[8]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[8]],na.rm=T),sd=sqrt(var(dd1[,v1[8]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[8]],na.rm=T),sd=sqrt(var(dd1[,v1[8]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,9]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[8,3]=parameter(vec,vec)[1]
PARAMETER1[8,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 9-------------------------------------------
lon<-LOC[v1[9],1]
lat<-LOC[v1[9],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[9]],na.rm=T),sd=sqrt(var(dd1[,v1[9]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[9]],na.rm=T),sd=sqrt(var(dd1[,v1[9]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[9]],na.rm=T),sd=sqrt(var(dd1[,v1[9]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,10]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[9,3]=parameter(vec,vec)[1]
PARAMETER1[9,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 10-------------------------------------------
lon<-LOC[v1[10],1]
lat<-LOC[v1[10],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[10]],na.rm=T),sd=sqrt(var(dd1[,v1[10]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[10]],na.rm=T),sd=sqrt(var(dd1[,v1[10]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[10]],na.rm=T),sd=sqrt(var(dd1[,v1[10]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,11]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[10,3]=parameter(vec,vec)[1]
PARAMETER1[10,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(2,7,12,13,14,15,16,17,25,35)])
testdf1<-union_1234[find_index(0,1,1,0),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4","o5","o6","o7","o8","o9","o10")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)),
                col5=rep(NA,nrow(testdf1)),col6=rep(NA,nrow(testdf1)),col7=rep(NA,nrow(testdf1)),col8=rep(NA,nrow(testdf1)),
                col9=rep(NA,nrow(testdf1)),col10=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])+(kkk[i,5]*ob.pdf[,6])+(kkk[i,6]*ob.pdf[,7])+
    (kkk[i,7]*ob.pdf[,8])+(kkk[i,8]*ob.pdf[,9])+(kkk[i,9]*ob.pdf[,10])+(kkk[i,10]*ob.pdf[,11])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)),col5=rep(NA,length(v1)),col6=rep(NA,length(v1)),
                 col7=rep(NA,length(v1)),col8=rep(NA,length(v1)),col9=rep(NA,length(v1)),
                 col10=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])+(kkk1[i,5]*ob.pdf[,6])+
    (kkk1[i,6]*ob.pdf[,7])+(kkk1[i,7]*ob.pdf[,8])+(kkk1[i,8]*ob.pdf[,9])+(kkk1[i,9]*ob.pdf[,10])+(kkk1[i,10]*ob.pdf[,11])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 


# -----------------------FOR testdf7 (8,9,10,14,16,17,18,23,24,27,29,32,34)------------------------------------------------------
v1<-c(8,9,10,14,16,17,18,23,24,27,29,32,34)
# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[v1],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))
LOC5<-rep(NA,length(vec))
LOC6<-rep(NA,length(vec))
LOC7<-rep(NA,length(vec))
LOC8<-rep(NA,length(vec))
LOC9<-rep(NA,length(vec))
LOC10<-rep(NA,length(vec))
LOC11<-rep(NA,length(vec))
LOC12<-rep(NA,length(vec))
LOC13<-rep(NA,length(vec))

DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4,LOC5,LOC6,LOC7,LOC8,LOC9,LOC10,LOC11,LOC12,LOC13)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 5-------------------------------------------
lon<-LOC[v1[5],1]
lat<-LOC[v1[5],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,6]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[5,3]=parameter(vec,vec)[1]
PARAMETER1[5,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 6-------------------------------------------
lon<-LOC[v1[6],1]
lat<-LOC[v1[6],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,7]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[6,3]=parameter(vec,vec)[1]
PARAMETER1[6,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 7-------------------------------------------
lon<-LOC[v1[7],1]
lat<-LOC[v1[7],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[7]],na.rm=T),sd=sqrt(var(dd1[,v1[7]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[7]],na.rm=T),sd=sqrt(var(dd1[,v1[7]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[7]],na.rm=T),sd=sqrt(var(dd1[,v1[7]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,8]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[7,3]=parameter(vec,vec)[1]
PARAMETER1[7,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 8-------------------------------------------
lon<-LOC[v1[8],1]
lat<-LOC[v1[8],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[8]],na.rm=T),sd=sqrt(var(dd1[,v1[8]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[8]],na.rm=T),sd=sqrt(var(dd1[,v1[8]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[8]],na.rm=T),sd=sqrt(var(dd1[,v1[8]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,9]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[8,3]=parameter(vec,vec)[1]
PARAMETER1[8,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 9-------------------------------------------
lon<-LOC[v1[9],1]
lat<-LOC[v1[9],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[9]],na.rm=T),sd=sqrt(var(dd1[,v1[9]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[9]],na.rm=T),sd=sqrt(var(dd1[,v1[9]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[9]],na.rm=T),sd=sqrt(var(dd1[,v1[9]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,10]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[9,3]=parameter(vec,vec)[1]
PARAMETER1[9,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 10-------------------------------------------
lon<-LOC[v1[10],1]
lat<-LOC[v1[10],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[10]],na.rm=T),sd=sqrt(var(dd1[,v1[10]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[10]],na.rm=T),sd=sqrt(var(dd1[,v1[10]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[10]],na.rm=T),sd=sqrt(var(dd1[,v1[10]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,11]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[10,3]=parameter(vec,vec)[1]
PARAMETER1[10,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

# ----------------------------------------FOR LOCATION 11-------------------------------------------
lon<-LOC[v1[11],1]
lat<-LOC[v1[11],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[11]],na.rm=T),sd=sqrt(var(dd1[,v1[11]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[11]],na.rm=T),sd=sqrt(var(dd1[,v1[11]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[11]],na.rm=T),sd=sqrt(var(dd1[,v1[11]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,12]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[11,3]=parameter(vec,vec)[1]
PARAMETER1[11,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

# ----------------------------------------FOR LOCATION 12-------------------------------------------
lon<-LOC[v1[12],1]
lat<-LOC[v1[12],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[12]],na.rm=T),sd=sqrt(var(dd1[,v1[12]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[12]],na.rm=T),sd=sqrt(var(dd1[,v1[12]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[12]],na.rm=T),sd=sqrt(var(dd1[,v1[12]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,13]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[12,3]=parameter(vec,vec)[1]
PARAMETER1[12,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 13-------------------------------------------
lon<-LOC[v1[13],1]
lat<-LOC[v1[13],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[13]],na.rm=T),sd=sqrt(var(dd1[,v1[13]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[13]],na.rm=T),sd=sqrt(var(dd1[,v1[13]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[13]],na.rm=T),sd=sqrt(var(dd1[,v1[13]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,14]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[13,3]=parameter(vec,vec)[1]
PARAMETER1[13,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(8,9,10,14,16,17,18,23,24,27,29,32,34)])
testdf1<-union_1234[find_index(0,1,1,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4","o5","o6","o7","o8","o9","o10","o11","o12","o13")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)),
                col5=rep(NA,nrow(testdf1)),col6=rep(NA,nrow(testdf1)),col7=rep(NA,nrow(testdf1)),col8=rep(NA,nrow(testdf1)),
                col9=rep(NA,nrow(testdf1)),col10=rep(NA,nrow(testdf1)),col11=rep(NA,nrow(testdf1)),col12=rep(NA,nrow(testdf1)),
                col13=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])+(kkk[i,5]*ob.pdf[,6])+(kkk[i,6]*ob.pdf[,7])+
    (kkk[i,7]*ob.pdf[,8])+(kkk[i,8]*ob.pdf[,9])+(kkk[i,9]*ob.pdf[,10])+(kkk[i,10]*ob.pdf[,11])+(kkk[i,11]*ob.pdf[,12])+(kkk[i,12]*ob.pdf[,13])+
    (kkk[i,13]*ob.pdf[,14])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)),col5=rep(NA,length(v1)),col6=rep(NA,length(v1)),
                 col7=rep(NA,length(v1)),col8=rep(NA,length(v1)),col9=rep(NA,length(v1)),
                 col10=rep(NA,length(v1)),col11=rep(NA,length(v1)),col12=rep(NA,length(v1)),col13=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])+(kkk1[i,5]*ob.pdf[,6])+
    (kkk1[i,6]*ob.pdf[,7])+(kkk1[i,7]*ob.pdf[,8])+(kkk1[i,8]*ob.pdf[,9])+(kkk1[i,9]*ob.pdf[,10])+(kkk1[i,10]*ob.pdf[,11])+
    (kkk1[i,11]*ob.pdf[,12])+(kkk1[i,12]*ob.pdf[,13])+(kkk1[i,12]*ob.pdf[,13])+(kkk1[i,13]*ob.pdf[,14])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -----------------------FOR testdf8 (1,5,20,29)------------------------------------------------------
v1<-c(1,5,21,30)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(1,5,21,30)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(1,5,21,30)])
testdf1<-union_1234[find_index(1,0,0,0),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),
                 col4=rep(NA,nrow(testdf1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -------------------------------------------------------------------------------
# -----------------------FOR testdf9 (5,19,20,30)------------------------------------------------------
v1<-c(5,19,20,30)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(5,19,20,30)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(5,19,20,30)])
testdf1<-union_1234[find_index(1,0,0,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),
                 col4=rep(NA,nrow(testdf1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# ---------------------------------NO DF FOR 10,11----------------------------------------------
# -------------------------------------------------------------------------------
# -----------------------FOR testdf12 (1,6,26,33)------------------------------------------------------
v1<-c(1,6,26,33)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(1,6,26,33)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(1,6,26,33)])
testdf1<-union_1234[find_index(1,1,0,0),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),
                 col4=rep(NA,nrow(testdf1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -----------------------FOR testdf13 (19,30,36,37)------------------------------------------------------
v1<-c(19,30,36,37)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(19,30,36,37)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(19,30,36,37)])
testdf1<-union_1234[find_index(1,1,0,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),
                 col4=rep(NA,nrow(testdf1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -----------------------FOR testdf14 (3,13,33)------------------------------------------------------
v1<-c(3,13,33)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[c(3,13,33)],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))


DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------
vvv<-as.vector(X_omega[c(3,13,33)])
testdf1<-union_1234[find_index(1,1,1,0),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 

# -------------------------------------------------------------------------------
# -----------------------FOR testdf15 (3,22,27,28,31,37)------------------------------------------------------
v1<-c(3,22,27,28,31,37)

# POINTS CONSIDERED
plot(data.shape)
plot(Delhi_int_circles[,"CC_1"],add=T)
plot(xy1[v1],add=T,col=rep(1:37, each = 1),pch=20)
legend("bottomleft", legend = paste("LOC", 1:37), col = 1:37, pch =20, bty = "n")

LOC1<-rep(NA,length(vec))
LOC2<-rep(NA,length(vec))
LOC3<-rep(NA,length(vec))
LOC4<-rep(NA,length(vec))
LOC5<-rep(NA,length(vec))
LOC6<-rep(NA,length(vec))

DATA1<-data.frame(x=seq(0,600,1),LOC1,LOC2,LOC3,LOC4,LOC5,LOC6)

dim<-ncol(DATA1)-1
PARAMETER1<-data.frame(x=LOC[v1,1],y=LOC[v1,2],shape=rep(NA,dim),scale=rep(NA,dim))


lon<-LOC[v1[1],1]
lat<-LOC[v1[1],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[1]],na.rm=T),sd=sqrt(var(dd1[,v1[1]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,2]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[1,3]=parameter(vec,vec)[1]
PARAMETER1[1,4]=parameter(vec,vec)[2]

# ----------------------------------------FOR LOCATION 2-------------------------------------------
lon<-LOC[v1[2],1]
lat<-LOC[v1[2],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[2]],na.rm=T),sd=sqrt(var(dd1[,v1[2]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,3]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[2,3]=parameter(vec,vec)[1]
PARAMETER1[2,4]=parameter(vec,vec)[2]



# ----------------------------------------FOR LOCATION 3-------------------------------------------
lon<-LOC[v1[3],1]
lat<-LOC[v1[3],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[3]],na.rm=T),sd=sqrt(var(dd1[,v1[3]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,4]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[3,3]=parameter(vec,vec)[1]
PARAMETER1[3,4]=parameter(vec,vec)[2]
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------FOR LOCATION 4-------------------------------------------
lon<-LOC[v1[4],1]
lat<-LOC[v1[4],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[4]],na.rm=T),sd=sqrt(var(dd1[,v1[4]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,5]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[4,3]=parameter(vec,vec)[1]
PARAMETER1[4,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 5-------------------------------------------
lon<-LOC[v1[5],1]
lat<-LOC[v1[5],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[5]],na.rm=T),sd=sqrt(var(dd1[,v1[5]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,6]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[5,3]=parameter(vec,vec)[1]
PARAMETER1[5,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------
# ----------------------------------------FOR LOCATION 6-------------------------------------------
lon<-LOC[v1[6],1]
lat<-LOC[v1[6],2]

v<-punif(lon, min =min(x2)-0.001, max = max(x2)+0.001)
w<-punif(lat, min =min(x3)-0.001, max = max(x3)+0.001)
# PDF OF FRANK
c_23<-VineCopula::BiCopPDF(v,w,selectedcopula1)

# PARAMETERS FOR CONDITIONAL PDF (PM2.5|(LONG,LAT))
theta1<-param
k1<-(1+theta1)*(1+(2*theta1))
k2<-theta1+1
term_3<-(v*w)^(-k2)
cons<-(1/c_23)*k1*term_3

k3<-(1+(3*theta1))/theta1
z<-(1/(v^theta1))+(1/(w^theta1))-1

likelihood.cop<-function(x,support){
  f<-function(x){
    g=(1+z*(pnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T)))))^(-k3)          
    h=(pnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T)))^(k3-k2))*(dnorm(x,mean =mean(dd1[,v1[6]],na.rm=T),sd=sqrt(var(dd1[,v1[6]],na.rm=T))))
    return(g*h*cons)
  }
  c<-1/sum(f(support))
  return(c*f(x))
}
DATA1[,7]=likelihood.cop(vec,vec)

parameter<-function(x,support){
  shape.par<-sum(vec*likelihood.cop(x,support))
  scale.par<-sqrt(sum(vec^(2)*likelihood.cop(vec,vec))-(shape.par)^2)
  return(list(shape.par,scale.par))
}
parameter(vec,vec)
PARAMETER1[6,3]=parameter(vec,vec)[1]
PARAMETER1[6,4]=parameter(vec,vec)[2]
# ------------------------------------------------------------------------------

vvv<-as.vector(X_omega[c(3,22,27,28,31,37)])
testdf1<-union_1234[find_index(1,1,1,1),]


# ---------------------------------------------------------------------------------
DATA_C4 <- data.frame(LONG= rep(NA,length(v1)),LAT=rep(NA,length(v1)))
for (i in 1:length(v1))
{
  DATA_C4[i,1]=LOC[v1[i],1]
  DATA_C4[i,2]=LOC[v1[i],2]
}

# CENTROID/RANDOM POINT AGGREGATED  MATRIX
Centroid_agg_DF_C4<-data.frame(LONG=DATA_C4$LONG,LAT=DATA_C4$LAT)
rownames(Centroid_agg_DF_C4) <- c("O1","O2","o3","o4","o5","o6")

w<-function(lat,lon,df){
  df.new<-as.data.frame(rbind(c(lat,lon),df))
  distance<-as.matrix(dist(df.new))[-1,]
  return(distance[,1]/sum(distance[,1]))
}
kkk<-data.frame(col1=rep(NA,nrow(testdf1)),col2=rep(NA,nrow(testdf1)),col3=rep(NA,nrow(testdf1)),col4=rep(NA,nrow(testdf1)),
                col5=rep(NA,nrow(testdf1)),col6=rep(NA,nrow(testdf1)))
df_points<-testdf1
# kkk is the weight matrix of all points in testdf
for(i in 1:nrow(df_points))
{
  kkk[i,]=w(df_points[i,1],df_points[i,2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist1<-list()
for(i in 1:nrow(df_points))
{
  datalist1[[i]]=(kkk[i,1]*ob.pdf[,2])+(kkk[i,2]*ob.pdf[,3])+(kkk[i,3]*ob.pdf[,4])+(kkk[i,4]*ob.pdf[,5])+(kkk[i,5]*ob.pdf[,6])+(kkk[i,6]*ob.pdf[,7])
    
}
pdf.new<-do.call(rbind,datalist1)
pdf.new<-as.data.frame(pdf.new)

datalist2<-list()

for(i in 1:nrow(df_points))
{
  datalist2[[i]]=pdf.new[i,]*(1/sum(pdf.new[i,]))
}
pdf.un<-do.call(rbind,datalist2)
pdf.un<-as.data.frame(pdf.un)

estimate.value<-vector()
for(i in 1:nrow(df_points))
{
  estimate.value[i]<-sum(ob.pdf[,1]*pdf.un[i,])
}
# INTERPOLATION CHECK
interdf1<-as.data.frame(cbind(testdf1,estimate.value))
library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
FINAL<-rbind(FINAL,interdf1)

PM2.5sf <- st_as_sf(x =interdf1,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="lightgrey",border="darkgrey") 

tm_shape(W) +tm_polygons()+  tm_shape(P) +
  tm_dots(col="estimate.value",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)

# --------------------------------RMSE CHECK-----------------------------------------------------
kkk1<-data.frame(col1=rep(NA,length(v1)),col2=rep(NA,length(v1)),col3=rep(NA,length(v1)),
                 col4=rep(NA,length(v1)),col5=rep(NA,length(v1)),col6=rep(NA,length(v1)))

for(i in 1:length(v1))
{
  kkk1[i,]=w(LOC[v1[i],1],LOC[v1[i],2],Centroid_agg_DF_C4)
}
ob.pdf<-DATA1
datalist3<-list()
for(i in 1:length(v1))
{
  datalist3[[i]]=(kkk1[i,1]*ob.pdf[,2])+(kkk1[i,2]*ob.pdf[,3])+(kkk1[i,3]*ob.pdf[,4])+(kkk1[i,4]*ob.pdf[,5])+(kkk1[i,5]*ob.pdf[,6])
}
pdf.new1<-do.call(rbind,datalist3)
pdf.new1<-as.data.frame(pdf.new1)

datalist4<-list()

for(i in 1:length(v1))
{
  datalist4[[i]]=pdf.new1[i,]*(1/sum(pdf.new1[i,]))
}
pdf.un1<-do.call(rbind,datalist4)
pdf.un1<-as.data.frame(pdf.un1)

estimate.value1<-vector()
for(i in 1:length(v1))
{
  estimate.value1[i]<-sum(ob.pdf[,1]*pdf.un1[i,])
}

cor(vvv,estimate.value1)^2
Metrics::rmse(vvv,estimate.value1) 


# --------------------------------FINAL PLOT-----------------------------------------------

library(sp)
library(tmap)
gadm <- readRDS("D:/Data Set of Delhi Air Pollution/gadm36_IND_1_sp.rds")
gadm
# ADDED COL
FINAL$abc<-FINAL$estimate.value/2.367
FINAL<-df

PM2.5sf <- st_as_sf(x =FINAL,coords = c("x1", "x2"),crs =crs(gadm))
P <- as(PM2.5sf, "Spatial")
W<-gadm[25,]
P@bbox <- W@bbox
# Shapefile for DELHI
plot(gadm[25,],col="white",border="black") 

#tm_shape(W) +tm_polygons()+ 
  tm_shape(P)+tm_dots(col="abc",shape=4,auto.palette.mapping = FALSE,stretch.palette=TRUE,
          title="Sampled PM2.5", size=0.7) +
  tm_legend(legend.outside=TRUE)
# -------------------------------------------------------------------

write.csv(FINAL, file = "IP_NOV_2019.csv")
  
dt<-read.csv("D:/Data Set of Delhi Air Pollution/IP_NOV_2019.csv")
head(dt)  
library(ggplot2)
colnames(d)<-c("x1","x2")
p
library(ggplot2)
v=ggplot(dt, aes(x = x1, y = x2)) 
v+geom_raster(aes(fill=X_omega),interpolate = T)+
v+  geom_tile(aes(fill=abc),colour="NA",alpha = 0.8)+ geom_point(d,mapping=aes(x=x1,y=x2,z=NULL),colour=1:38,pch=7)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  