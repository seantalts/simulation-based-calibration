library(bayesplot)
library(ggplot2)

ppc_ecdf_overlay_2 <- function (y, yrep, ..., pad = TRUE, size = 0.25, alpha = 0.7) 
{
  
  y <- bayesplot:::validate_y(y)
  yrep <- bayesplot:::validate_yrep(yrep, y)
  ggplot(bayesplot:::melt_yrep(yrep), aes_(x = ~value)) + hline_at(c(0, 
                                                         0.5, 1), size = c(0.2, 0.1, 0.2), linetype = 2, color = bayesplot:::get_color("dh")) + 
    stat_ecdf(mapping = aes_(group = ~rep_id, color = "yrep"), 
              geom = "step", size = size, alpha = alpha, pad = pad) + 
    stat_ecdf(data = data.frame(value = y), mapping = aes_(color = "y"), 
              geom = c("step"), size = 1, pad = pad) + bayesplot:::scale_color_ppc_dist() + 
    xlab(bayesplot:::y_label()) + 
    scale_x_continuous(limits=c(0,100),expand=c(0,0)) + scale_y_continuous(limits=c(0,1),expand=c(0,0),breaks = c(0, 0.5, 
                                                    1)) + yaxis_title(FALSE) + xaxis_title(FALSE) + yaxis_ticks(FALSE)
}


load(file = "output_sbc_inla.RData")


for(i in  1:5) {
  fname = paste("adm",i,".samples",sep="")
  write(rrr[,i],ncolumns = 1,file = fname)
  
  samps = matrix(sample(c(1:100),size = 1000*500, replace=T),500,1000)
  ppc_ecdf_overlay_2(rrr[,i],samps) + geom_abline(slope=1/100,intercept=0,colour="grey45",linetype="dashed")
  fname2=paste("adm",i,"_ecdf.eps",sep="")
  ggsave(filename = fname2,device = "cairo_ps",family="serif")
  
  
  
}


if(FALSE) {
  #ggplot version of the gnuplot histograms

  rr = data.frame(adm11 = rrr[,1])
  n_breaks=101
  
  #approximate CI (slightly conservative)
  mean=1000/n_breaks
  sd = sqrt(mean)
  ggplot(rr,aes(x=adm11))  + geom_segment(aes(x=0,y=mean,xend=100,yend=mean),colour="grey25") + 
    geom_polygon(data=data.frame(x=c(-5,0,-5,105,100,105,-5),y=c(mean-3*sd,mean,mean+3*sd,mean+3*sd,mean,mean-3*sd,mean-3*sd)),aes(x=x,y=y),fill="grey45",color="grey25",alpha=0.5) +
    geom_histogram(breaks=seq(-1,100,by=100/n_breaks),fill="#A25050",colour="black")
  
  #exact CI
 CI = qbinom(c(0.005,0.5,0.995), size=1000,prob  =  1/101)

 ggplot(rr,aes(x=adm11))  + geom_segment(aes(x=0,y=mean,xend=100,yend=mean),colour="grey25") + 
   geom_polygon(data=data.frame(x=c(-5,0,-5,105,100,105,-5),y=c(CI[1],CI[2],CI[3],CI[3],CI[2],CI[1],CI[1])),aes(x=x,y=y),fill="grey45",color="grey25",alpha=0.5) +
   geom_histogram(breaks=seq(-1,100,by=100/n_breaks),fill="#A25050",colour="black")
 
  }

