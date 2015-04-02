from ggplot import *
def ggplotexample():
#	print diamonds.head()
	print meat.head()
	p = ggplot(aes(x='date',y='beef'),data=meat)
	print p + geom_point(color='blue')+theme_bw()+stat_smooth(color='red')
		
if __name__== "__main__":
	ggplotexample()
	