from fpdf  import FPDF, HTMLMixin

class MyFPDF(FPDF, HTMLMixin):
    pass

title = 'Shakespeare style transformer '

class PDF(FPDF):

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')


WIDTH = 210
HEIGHT = 297
pdf = MyFPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True)
pdf.image('logo.jpg', x=150, y=10, w=40, h=30)
pdf.image('shakespear.jpg', x=10, y=10, w=40, h=30)
pdf.set_font('Times', 'B', 18)
pdf.cell(0,10,txt = title, align='C', ln = 1)
pdf.ln(60)


############################################## Executive summary ##################################################################
pdf.set_font('Times', 'B', 14)
pdf.cell(0,10,txt = 'Executive summary and conclusions', align='C', ln = 1)
txt = '''       <font face = "Times" size = "12"><p line-height=2 > The aim of the task was to create a transformer language model which will generate a text with Shakespeare style. \
The model should get the length of desired text ( about 400 - 600 chars) and generate a text close to the length. Shakeseare dataset used in \
this report is from <a href=" https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays">here</a>. As a result of time and space constraints,we have \
decided to train the transformer model with 3 epochs which in turn led to high Crossentropy loss (0.7), so random guesses of \
the next word is better than using the below described transformer model. Also predicting the text with 400-600 characters isn't \
possible at the moment, as the algorithm is weak and after some text generation it says that there is \
higher probability that there are no other words. \
    So we can conclude that further analysis on a computer's memory efficient management during the training of a deep learning model \
and transformer training with more epoches can raise the performance of the model.</p></font>
'''
# pdf.set_font('Times', '', 12, )
pdf.write_html(txt)

pdf.ln(30)
############################################## Executive summary ##################################################################
pdf.set_font('Times', 'B', 14)
pdf.cell(0,10,txt = 'Exploratory data analysis', align='C', ln = 1)
txt = '''    As the aim of the project is to generate new words, we have used only the 'PlayerLine' column from the Shakespeare dataset, \
where we have 111.396 rows of strings. Without removing stop words and punctuation marks there are 27.381 unique tokens in the dataset \
and after the removal of mentioned tokens we have 27.245 unique words. Inspite of these facts we will set vocabulary size 2000 \
in our model. This has been done just for understanding the structure \
of the dataset, because in the transformer model we will also use stop words and punctuation marks to make generated text more \
similar to normal texts. Figure 1 below shows the word cloud of the Shakespearian vocabulary. 
'''


pdf.set_font('Times', '', 12, )
pdf.multi_cell(0,10,txt=txt)
pdf.set_font('Times', 'B', 10)
pdf.cell(0,10,txt = 'Figure 1. Wordcloud of Shakespearean vocabulary',  ln = 1)
pdf.image(name='../output/images/wordcloud.png',w=150, h=100, x=0)


pdf.ln(20)

##############################################Trandformer block ##################################################################
pdf.set_font('Times', 'B', 14)
pdf.cell(0,10,txt = 'Transformer language model', align='C', ln = 1)
txt = '''   For model creation these hyperparameters have been used: vocab_size = 20000, batch_size = 64, \
maxlen = 50, embed_dim = 128, num_heads = 2, feed_forward_dim = 256 and num_transformer_blocks = 2. Model's \
structure is represented below. 
'''
pdf.set_font('Times', '', 12, )
pdf.multi_cell(0,10,txt=txt)
pdf.set_font('Times', 'B', 10)
pdf.cell(0,10,txt = 'Figure 2. Model structure',  ln = 1)
pdf.image(name='../output/images/model_structure.png',w=120, h=90)


txt = """ The summary of the trained model is shown below. In Figure 3 cross-entropy loss for each epoch is represented.
"""
pdf.set_font('Times', '', 12, )
pdf.multi_cell(0,10,txt=txt)

with open('../output/textgen_model_summary.txt', 'rb') as fh:
    txt = fh.read().decode('latin-1')
    # Times 12
pdf.set_font('Times', '', 12)
# Output justified text
pdf.set_font('Times', 'B', 10)
pdf.cell(0,10,txt = "Summary: Transformer model's summary",  ln = 1)
pdf.multi_cell(0, 5, txt)


pdf.set_font('Times', 'B', 10)
pdf.cell(0,10,txt = 'Figure 3. Cross-entropy loss for each epoch ',  ln = 1)
pdf.image(name='../output/images/loss.png',w=130, h=110, x=0)


txt = """    The model was trained for 2.5 hours(see log_history.txt). During each epoch we generated text to see how the \
model is improving after each epoch. After the first epoch we have the sentence 'to be so much but i am so so much but i am so' and \
after the second epoch it changed to 'to be so much but i am so'. After training the model we tried to use the positive part ('To be') \
from the most famous citation of Shakespeare ("To be, or not to be: that is the question.") and our transformer gave us \
equally rhetorical answer 'and the king is well the prince i must be a'.

"""


pdf.set_font('Times', '', 12, )
pdf.multi_cell(0,10,txt=txt)

# generating the report
pdf.output('Shakespeare_custom_trans.pdf', 'F')