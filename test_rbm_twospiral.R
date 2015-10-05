## Test RBM with two spiral dataset

library(stats)
library(ggplot2)

source("rbm.R")

##' Generate two spiral test data
##'
##' @param n Number of points for EACH spiral
##' @param len Length of the spiral, which is rad / 2pi.
##' @param start Ratio of starting rad / 2pi.
##' @param noise Noise level.
##' @param pp T if preprocessed
two_spiral <- function(n=100, len=1, start=0.02, noise=0, pp=T) {

  t <- (start + sqrt(runif(n)) * len) * pi * 2

  a <- matrix(c(-cos(t) * t + runif(n) * noise,
                sin(t) * t + runif(n) * noise),
              ncol=2)
  b <- matrix(c(cos(t) * t + runif(n) * noise,
                -sin(t) * t + runif(n) * noise),
              ncol=2)

  colnames(a) <- c("x", "y")
  data <- rbind(a, b)

  ## Scale data into an unit circle centered at origin.
  scale <- max(rowSums(data * data))
  data <- data / sqrt(scale)

  ## Move and scale the unit circle to 0-1.
  data <- (data + 1) / 2

  list(data = data, label = c(rep(0, n), rep(1, n)))
}

## Test Parameters
## -------------------------------------------------------------------

## Number of training samples each category.
n <- 100

## Number of random samples.
m <- 1000

## Number of visible units and hidden units.
nvis <- 2
nhid <- 100

batch.size <- 10
max.epoch <- 1000

## Learning rate.
lr <- 0.1

## Number of Gibbs steps during training.
k <- 1

## Use PCD-k if persistent, CD-k otherwise.
persistent <- F

## Collect samples sample.n times, after plot.every steps of Gibbs
## sampling.
plot.every <- 1
sample.n <- 100
sample.point <- c(2, 10, 50, 100)

## Test Code
## -------------------------------------------------------------------

twospiral <- two_spiral(n, len=0.5, start=0.1, noise=1)

## The original data
p <- ggplot(data.frame(twospiral$data), aes(x=x, y=y))
p <- p + geom_point(aes(colour=factor(twospiral$label)))
p <- p + scale_colour_manual(values = c("red", "blue"))
p <- p + xlim(0, 1) + ylim(0, 1)
p <- p + theme(axis.title.x=element_blank(),
               axis.title.y=element_blank())
p <- p + guides(colour=F)

pdf("img/twospiral-origin.pdf")
print(p)
dev.off()

svg("img/twospiral-origin.svg")
print(p)
dev.off()

## First train the rbm.
rbm <- rbm_setup(nvis=nvis, nhid=nhid)

## See what happens before training

dat.cons <- list()
dat.cons <- c(dat.cons, list(twospiral$data))

print("Sampling...")
tmp <- twospiral$data
for (i in 1:sample.n) {
  for (j in 1:plot.every) {
    cat(sprintf("Sampling step: %5d\r", j))
    tmp <- rbm_gibbs_vhv(rbm, tmp)$E
  }
  cat(sprintf("\nDone %2dth sample cycle\n", i))
  colnames(tmp) <- c("x", "y")
  dat.cons <- c(dat.cons, list(tmp))
}

cat("\nDone sampling before training.\n")

for (i in 2:(sample.n + 1)) {
  df <- data.frame(rbind(dat.cons[[i - 1]], dat.cons[[i]]))
  df$grp1 <- c(twospiral$label, twospiral$label)
  df$grp2 <- c(1:(2 * n), 1:(2 * n))

  p <- ggplot(df, aes(x=x, y=y))
  p <- p + geom_point(aes(colour=factor(grp1)))
  p <- p + scale_colour_manual(values = c("red", "blue"))
  p <- p + geom_line(aes(group=factor(grp2)), colour="green", alpha=0.2)
  p <- p + xlim(0, 1) + ylim(0, 1)
  p <- p + guides(colour=F)
  p <- p + theme(axis.title.x=element_blank(),
                 axis.title.y=element_blank())

  png(sprintf("img/twospiral-pretrain-%03d.png", i), width=640, height=640)
  print(p)
  dev.off()

  if (i %in% sample.point) {
    pdf(sprintf("img/twospiral-pretrain-%03d.pdf", i))
    print(p)
    dev.off()

    svg(sprintf("img/twospiral-pretrain-%03d.svg", i))
    print(p)
    dev.off()
  }
}

## See what happens after training

print("Training...")
rbm <- rbm_train(rbm, twospiral$data, lr=lr, max.epoch=max.epoch, k=k,
                 batch.size=batch.size, persistent=persistent)

## Now sample from the rbm staring from the original data

dat.cons <- list()
dat.cons <- c(dat.cons, list(twospiral$data))

print("Sampling...")
tmp <- twospiral$data
for (i in 1:sample.n) {
  for (j in 1:plot.every) {
    cat(sprintf("Sampling step: %5d\r", j))
    tmp <- rbm_gibbs_vhv(rbm, tmp)$E
  }
  cat(sprintf("\nDone %2dth sample cycle\n", i))
  colnames(tmp) <- c("x", "y")
  dat.cons <- c(dat.cons, list(tmp))
}

cat("\nDone sampling after training.\n")

for (i in 2:(sample.n + 1)) {
  df <- data.frame(rbind(dat.cons[[i - 1]], dat.cons[[i]]))
  df$grp1 <- c(twospiral$label, twospiral$label)
  df$grp2 <- c(1:(2 * n), 1:(2 * n))

  p <- ggplot(df, aes(x=x, y=y))
  p <- p + geom_point(aes(colour=factor(grp1)))
  p <- p + scale_colour_manual(values = c("red", "blue"))
  p <- p + geom_line(aes(group=factor(grp2)), colour="green", alpha=0.2)
  p <- p + xlim(0, 1) + ylim(0, 1)
  p <- p + guides(colour=F)
  p <- p + theme(axis.title.x=element_blank(),
                 axis.title.y=element_blank())

  png(sprintf("img/twospiral-cons-%03d.png", i), width=640, height=640)
  print(p)
  dev.off()

  if (i %in% sample.point) {
    pdf(sprintf("img/twospiral-cons-%03d.pdf", i))
    print(p)
    dev.off()

    svg(sprintf("img/twospiral-cons-%03d.svg", i))
    print(p)
    dev.off()
  }
}
