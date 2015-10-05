## Test RBM with MNIST dataset

library(ggplot2)
library(grid)

source("rbm.R")

##' Load MNIST dataset
##'
##' @param d The directory containing MNIST dataset.
##' @param pp T to preprocess.
##' @return A list containing training and test data.
mnist_load <- function(d, pp=T)
{
  load_helper <- function(id)
  {
    ret <- list()

    ## Read in image data
    f <- file(sprintf("%s/%s-images-idx3-ubyte", d, id), "rb")

    ## magic number
    readBin(f, "integer", n=1, size=4, endian="big")

    ## number of images
    ret$n <- readBin(f, "integer", n=1, size=4, endian="big")

    ret$nrow <- readBin(f, "integer", n=1, size=4, endian="big")
    ret$ncol <- readBin(f, "integer", n=1, size=4, endian="big")

    x <- readBin(f, "integer", n=ret$n * ret$nrow * ret$ncol,
                 size=1, signed=F)
    ret$data <- matrix(x, ncol=ret$nrow * ret$ncol, byrow=T)
    close(f)

    ## Read in label data
    f <- file(sprintf("%s/%s-labels-idx1-ubyte", d, id), "rb")
    readBin(f, "integer", n=2, size=4, endian="big")
    ret$label <- readBin(f, "integer", n=ret$n, size=1, signed=F)
    close(f)

    ## Finish loading data, now preprocessing the data if asked for.
    if (pp) {
      ret$data <- ret$data / 255
    }

    ret
  }

  list(
    train=load_helper("train"),
    test=load_helper("t10k")
  )
}

mnist_print_rnd <- function(data, n=10) {
  samples <- by(data$data, data$label, function(x) x[sample.int(nrow(x), n),])

  samples <- lapply(samples, function(x) {
    x <- as.matrix(x)
    do.call(rbind, lapply(split(x, row(x)),
                          matrix, nrow=28, ncol=28, byrow=T))
  })

  samples <- do.call(cbind, samples)

  pdf("img/mnist-rnd.pdf")
  grid.raster(samples)
  dev.off()

  svg("img/mnist-rnf.svg")
  grid.raster(samples)
  dev.off()
}

## Test parameter
## -------------------------------------------------------------------

data.dir <- "data"

cat("Loading data...")
## We are testing generative model, so test set is not required.
mnist <- mnist_load(data.dir)$train
print("Done")

## Number of visible and hidden units
nvis <- mnist$nrow * mnist$ncol
nhid <- 500

batch.size <- 20
max.epoch <- 15

## Learning rate.
lr <- 0.1

## T to use PCD-k, CD-k otherwise.
persistent <- T

## Gibbs sampling step in training
k <- 1

## Collect samples sample.n times, after plot.every steps of Gibbs
## sampling.
plot.every <- 1000
sample.n <- 9

## Test
## -------------------------------------------------------------------

mnist_print_rnd(mnist)

rbm <- rbm_setup(nvis=nvis, nhid=nhid)

print("Training...")

tm <- system.time({
  rbm <- rbm_train(rbm, mnist$data, lr=lr, max.epoch=max.epoch, k=k,
                   batch.size=batch.size, persistent=persistent)
})

print(tm)

dat.orig <- by(mnist$data, mnist$label, function(x) x[sample.int(nrow(x), 1),])
dat.orig <- lapply(dat.orig, as.matrix)
dat.orig <- do.call(rbind, dat.orig)

dat.cons <- list()
dat.cons <- c(dat.cons, list(dat.orig))

print("Sampling...")
tmp <- dat.orig
for (i in 1:sample.n) {
  for (j in 1:plot.every) {
    cat(sprintf("Sampling step: %5d\r", j))
    tmp <- rbm_gibbs_vhv(rbm, tmp)$E
  }
  cat(sprintf("\nDone %2dth sample cycle\n", i))
  dat.cons <- c(dat.cons, list(tmp))
}

cat("Done sampling\n")

img.cons <- lapply(dat.cons, function(x) {
  do.call(cbind, lapply(split(x, row(x)),
                        matrix, nrow=28, ncol=28, byrow=T))
})

img.cons <- do.call(rbind, img.cons)

pdf("img/mnist-recons.pdf")
grid.raster(img.cons)
dev.off()

svg("img/mnist-recons.svg")
grid.raster(img.cons)
dev.off()
