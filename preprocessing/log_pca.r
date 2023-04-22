library(logisticPCA)
library(ggplot2)

dat = read.csv("../datasetutils/db/methods.csv", sep=";")
classes = dat$class
dat = dat[, !colnames(dat) %in% c("id", "class")]
dat = apply(dat, 2, function(x) as.numeric(x=="True"))
logpca_cv = cv.lpca(dat, ks = 2, ms = 1:10)
plot(logpca_cv)
logpca_model = logisticPCA(dat, k = 2, m = which.min(logpca_cv))
write.csv(predict(logpca_model, dat, type = "PCs"), "logpca_2dims.csv")

logsvd_model = logisticSVD(dat, k = 2)
write.csv(predict(logsvd_model, dat, type = "PCs"), "logsvd_2dims.csv")
# clogpca_model = convexLogisticPCA(dat, k = 2, m = which.min(logpca_cv))

plot(logsvd_model, type = "scores") + geom_point(aes(colour = classes)) + ggtitle("Exponential Family PCA") + scale_colour_manual(values = c("blue", "red", "green"))
plot(logpca_model, type = "scores") + geom_point(aes(colour = classes))  + ggtitle("Logistic PCA") + scale_colour_manual(values = c("blue", "red", "green"))