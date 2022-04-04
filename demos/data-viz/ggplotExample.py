import ggplot as g

def ggplotexample():
	print(meat.head())
	p = g.ggplot(aes(x='date',y='beef'),data=meat)
	print p + g.geom_point(color='blue') + g.theme_bw() + g.stat_smooth(color='red')
		
if __name__== "__main__":
	ggplotexample()	
