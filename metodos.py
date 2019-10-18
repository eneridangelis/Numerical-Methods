from sympy import*
y, t, x = symbols('y t x');
from mpmath import *
import matplotlib.pyplot as plt

vector_t = []
vector_y = []
vector_func_y = []
vector_y_am = []
vector_t_am = []
vector_y_fi = []
vector_t_fi = []


def comp(tn, yn, expfunc):
	return expfunc.subs([(t, tn),(y, yn)])

def show_graphic(vec_t, vec_y):
    plt.plot(vec_t, vec_y)
    plt.ylabel("y values")
    plt.xlabel("t values")
    plt.show()

############################ IMPLEMENTAÇÃO EULER ############################
def euler(y0, t0, h, quant, expfunc):
	vector_t.append(t0)
	vector_y.append(y0)

	for i in range(1, quant+1):
		func_ans = expfunc.subs([(t,t0), (y,y0)]) 
		y0 += h*func_ans
		t0 += h

		vector_t.append(t0)
		vector_y.append(y0)

	return t0

############################ IMPLEMENTAÇÃO EULER INVERSO ############################
def euler_inverso(y0, t0, h, quant, expfunc):
	vector_t.append(t0)
	vector_y.append(y0)

	for i in range(1, quant+1):
		## predĩção por euler ##
		func_ans = y0 + h * comp(t0, y0, expfunc)
		
		t0 += h
		y0 += h*comp(t0, func_ans, expfunc)
		
		# func_ans = solve(t1, y1, expfunc)
		# y0 += h*func_ans
		# t0 += h

		vector_t.append(t0)
		vector_y.append(y0)

	return t0

############################ IMPLEMENTAÇÃO EULER APRIMORADO ############################
def euler_aprimorado(y0, t0, h, quant, expfunc):
	vector_t.append(t0)
	vector_y.append(y0)

	for i in range(1, quant+1):
		y0 = y0 + (h/2) *(expfunc.subs([(t,t0), (y,y0)]) + comp(t0+h, y0+(expfunc.subs([(t,t0), (y,y0)]))*h, expfunc))
		t0 += h

		vector_t.append(t0)
		vector_y.append(y0)

	return t0

############################ IMPLEMENTAÇÃO RUNGE-KUTTA ############################
def runge_kutta(y0, t0, h, quant, expfunc):
	vector_t.append(t0)
	vector_y.append(y0)

	for i in range(1, quant+1):
		kn1 = expfunc.subs([(t,t0), (y,y0)])
		kn2 = expfunc.subs([(t,t0+(h/2)), (y,y0+((h/2)*kn1))])
		kn3 = expfunc.subs([(t,t0+(h/2)), (y,y0+((h/2)*kn2))])
		kn4 = expfunc.subs([(t,t0+h), (y,y0+(h*kn3))])

		y0 += h*((kn1+(2*kn2)+(2*kn3)+kn4)/6)
		t0 += h

		vector_t.append(t0)
		vector_y.append(y0)

	return t0

############################ IMPLEMENTAÇÃO ADAM-BASHFORTH ############################
def adam_bashforth(t0, h, quant, expfunc, order):
	# print(vector_y, vector_t, t0, quant, expfunc, order)
	y0 = vector_y[len(vector_y)-1]
	## ordem 2 ##
	if order == 2:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 3):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((3/2)*vector_func_y[0] - (1/2)*vector_func_y[1])
			t0 += h

			vector_y.append(y0)
	## ordem 3 ##
	elif order == 3:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 4):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((23/12)*vector_func_y[0] - (4/3)*vector_func_y[1] + (5/12)*vector_func_y[2])
			t0 += h

			vector_y.append(y0)
	## ordem 4 ##
	elif order == 4:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 5):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((55/24)*vector_func_y[0] - (59/24)*vector_func_y[1] + (37/24)*vector_func_y[2] - (3/8)*vector_func_y[3])
			t0 += h

			vector_y.append(y0)
	## ordem 5 ##
	elif order == 5:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 6):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((1901/720)*vector_func_y[0] - (1387/360)*vector_func_y[1] + (109/30)*vector_func_y[2] - (637/360)*vector_func_y[3] + (251/720)*vector_func_y[4])
			t0 += h

			vector_y.append(y0)
	## ordem 6 ##
	elif order == 6:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 7):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((4277/1440)*vector_func_y[0] - (2641/480)*vector_func_y[1] + (4991/720)*vector_func_y[2] - (3649/720)*vector_func_y[3] + (959/480)*vector_func_y[4] - (95/288)*vector_func_y[5])
			t0 += h

			vector_y.append(y0)
	## ordem 7 ##
	elif order == 7:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 8):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((198721/60480)*vector_func_y[0] - (18637/2520)*vector_func_y[1] + (235183/20160)*vector_func_y[2] - (10754/945)*vector_func_y[3] + (135713/20160)*vector_func_y[4] - (5603/2520)*vector_func_y[5] + (19087/60480)*vector_func_y[6])
			t0 += h

			vector_y.append(y0)
	## ordem 8 ##
	elif order == 8:
		for i in range(len((vector_y)),quant+1):
			vector_t.append(t0)
			vector_func_y.clear()
			for j in range(1, 9):
				vector_func_y.append(comp(vector_t[len(vector_y)-j], vector_y[len(vector_y)-j], expfunc))

			y0 += h*((16083/4480)*vector_func_y[0] - (1152169/120960)*vector_func_y[1] + (242653/13440)*vector_func_y[2] - (296053/13440)*vector_func_y[3] + (2102243/120960)*vector_func_y[4] - (115747/13440)*vector_func_y[5] + (32863/13440)*vector_func_y[6] - (5257/17280)*vector_func_y[7])
			t0 += h

			vector_y.append(y0)


############################ IMPLEMENTAÇÃO ADAM-MULTON ############################
def adam_multon(t0, h, quant, expfunc, order):
	y0 = vector_y_am[len(vector_y_am)-1]

	## ordem 2 ##
	if order == 2:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = comp(vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 2):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h/2*(vector_func_y[0] + func_n1)
			t0 += h

			vector_y_am.append(y0)

	## ordem 3 ##
	if order == 3:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = comp(vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 3):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h*((5/12)*func_n1 + (2/3)*vector_func_y[0] - (1/12)*vector_func_y[1])
			t0 += h

			vector_y_am.append(y0)

	## ordem 4 ##
	if order == 4:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = comp(vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 4):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h*((3/8)*func_n1 + (19/24)*vector_func_y[0] - (5/24)*vector_func_y[1] + (1/24)*vector_func_y[2])
			t0 += h

			vector_y_am.append(y0)

	## ordem 5 ##
	if order == 5:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = comp(vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 5):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h*((251/720)*func_n1 + (323/360)*vector_func_y[0] - (11/30)*vector_func_y[1] + (53/360)*vector_func_y[2] - (19/720)*vector_func_y[3])
			t0 += h

			vector_y_am.append(y0)

	## ordem 6 ##
	if order == 6:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = comp(vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 6):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h*((95/288)*func_n1 + (1427/1440)*vector_func_y[0] - (133/240)*vector_func_y[1] + (241/720)*vector_func_y[2] - (173/1440)*vector_func_y[3] + (3/160)*vector_func_y[4])
			t0 += h

			vector_y_am.append(y0)

	## ordem 7 ##
	if order == 7:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = comp(vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 7):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h*((19087/60480)*func_n1 + (2713/2520)*vector_func_y[0] - (15487/20160)*vector_func_y[1] + (586/945)*vector_func_y[2] - (5737/20160)*vector_func_y[3] + (263/2520)*vector_func_y[4] - (863/60480)*vector_func_y[5])
			t0 += h

			vector_y_am.append(y0)

	## ordem 8 ##
	if order == 8:
		for i in range(len(vector_y_am), quant+1):
			vector_t_am.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1) ##
			func_n1 = y0 + (vector_t_am[len(vector_t_am)-1], yn1, expfunc)

			for j in range(1, 8):
				vector_func_y.append(comp(vector_t_am[len(vector_y_am)-j], vector_y_am[len(vector_y_am)-j], expfunc))

			y0 += h*((5257/17280)*func_n1 + (139849/120960)*vector_func_y[0] - (4511/4480)*vector_func_y[1] + (123133/120960)*vector_func_y[2] - (88574/120960)*vector_func_y[3] + (1537/4480)*vector_func_y[4] - (11351/120960)*vector_func_y[5] + (275/24192)*vector_func_y[6])
			t0 += h

			vector_y_am.append(y0)

############################ IMPLEMENTAÇÃO FÓRMULA INVERSA ############################
def formula_inversa(t0, h, quant, expfunc, order):
	y0 = vector_y_fi[len(vector_y_fi)-1]
	# print(expfunc, order)
	## ordem 2 ##
	if order == 2:
		for i in range(len(vector_y_fi), quant+1):
			vector_t_fi.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1)
			func_n1 = comp(vector_t_fi[len(vector_t_fi)-1], yn1, expfunc)

			for j in range(1, 3):
				vector_func_y.append(vector_y_fi[len(vector_y_fi)-j])

			y0 = (4/3)*vector_func_y[0] - (1/3)*vector_func_y[1] + (2/3)*h*func_n1
			t0 += h

			vector_y_fi.append(y0)

	## ordem 3 ##
	if order == 3:
		for i in range(len(vector_y_fi), quant+1):
			vector_t_fi.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1)
			func_n1 = comp(vector_t_fi[len(vector_t_fi)-1], yn1, expfunc)

			for j in range(1, 4):
				vector_func_y.append(vector_y_fi[len(vector_y_fi)-j])

			y0 = (18/11)*vector_func_y[0] - (9/11)*vector_func_y[1] + (2/11)*vector_func_y[2] + (6/11)*h*func_n1
			t0 += h

			vector_y_fi.append(y0)

	## ordem 4 ##
	if order == 4:
		for i in range(len(vector_y_fi), quant+1):
			vector_t_fi.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1)
			func_n1 = comp(vector_t_fi[len(vector_t_fi)-1], yn1, expfunc)

			for j in range(1, 5):
				vector_func_y.append(vector_y_fi[len(vector_y_fi)-j])

			y0 = (48/25)*vector_func_y[0] - (36/25)*vector_func_y[1] + (16/25)*vector_func_y[2] - (3/25)*vector_func_y[3] + (12/25)*h*func_n1
			t0 += h

			vector_y_fi.append(y0)

	## ordem 5 ##
	if order == 5:
		for i in range(len(vector_y_fi), quant+1):
			vector_t_fi.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			## fazendo f(tn+1,yn+1)
			func_n1 = comp(vector_t_fi[len(vector_t_fi)-1], yn1, expfunc)

			for j in range(1, 6):
				vector_func_y.append(vector_y_fi[len(vector_y_fi)-j])

			y0 = (300/137)*vector_func_y[0] - (300/137)*vector_func_y[1] + (200/137)*vector_func_y[2] - (75/137)*vector_func_y[3] + (12/137)*vector_func_y[4] + (60/137)*h*func_n1
			t0 += h

			vector_y_fi.append(y0)

	## ordem 6 ##
	if order == 6:
		for i in range(len(vector_y_fi), quant+1):
			vector_t_fi.append(t0)
			vector_func_y.clear()

			## estimando yn+1 por Euler Inverso ##
			euler_inverso(y0, t0, h, 1, expfunc)
			yn1 = vector_y[len(vector_y)-1]

			# print(yn1)

			## fazendo f(tn+1,yn+1)
			func_n1 = comp(vector_t_fi[len(vector_t_fi)-1], yn1, expfunc)

			for j in range(1, 7):
				vector_func_y.append(vector_y_fi[len(vector_y_fi)-j])

			# print(vector_func_y)

			y0 = (360/147)*vector_func_y[0] - (450/147)*vector_func_y[1] + (400/147)*vector_func_y[2] - (225/147)*vector_func_y[3] + (72/147)*vector_func_y[4] - (10/147)*vector_func_y[5] + (60/147)*h*func_n1
			t0 += h

			vector_y_fi.append(y0)

	

############################ IMPLEMENTAÇÃO ADAMS-BASHFORTH POR EULER ############################
# def adam_bashforth_by_euler():




#################################### MAIN ####################################
def main():
	#f = open('saida.txt', 'w');

	f = open('entrada.txt', 'r')

	for line in f:
		vector_t.clear()
		vector_y.clear()
		vector_func_y.clear()

		inputf = line.split(' ')
		#print(inputf)
		if inputf[0] == 'euler' or inputf[0] == 'euler_inverso' or inputf[0] == 'euler_aprimorado' or inputf[0] == 'runge_kutta':
			y0 = inputf[1]
			t0 = inputf[2]
			h = inputf[3]
			quant = inputf[4]
			expin = inputf[5].split('\n')
			# expfunc = expin[0]
			# print(expfunc)
			expfunc = sympify(expin[0])
			# print(expfunc)
			# print(y0, t0, h, quant, expfunc)
			# print(expfunc.subs([(t,1),(y,2)]))			
			if inputf[0] == 'euler':
				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Euler' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				euler(float(y0), float(t0), float(h), int(quant), expfunc)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Euler")

			elif inputf[0] == 'euler_inverso':
				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Euler Inverso' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				euler_inverso(float(y0), float(t0), float(h), int(quant), expfunc)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Euler Inverso")

			elif inputf[0] == 'euler_aprimorado':
				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Euler Aprimorado' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				euler_aprimorado(float(y0), float(t0), float(h), int(quant), expfunc)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Euler Aprimorado")

			elif inputf[0] == 'runge_kutta':
				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Runge-Kutta' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				runge_kutta(float(y0), float(t0), float(h), int(quant), expfunc)
 
				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Runge-Kutta")	

			show_graphic(vector_t, vector_y)		

		elif inputf[0] == 'adam_bashforth':
			order = inputf[len(inputf)-1].split('\n')
			order = int(order[0])
			# print(order)
			for i in range(1, order+1):
				vector_y.append(float(inputf[i]))
			t0 = float(inputf[order+1])
			h = inputf[order+2]
			quant = inputf[order+3]
			expin = inputf[order+4].split('\n')
			expfunc = sympify(expin[0])


			f_out = open('saida.txt', 'a')
			f_out.write('Metodo de Adams-Bashforth' + '\n')
			f_out.write('y(' + str(t0) + ') = ' + str(vector_y[0]) + '\n')
			f_out.write('h = ' + h + '\n')

			
			# print(vector_y)
			for i in range(1, order+1):
				vector_t.append(float(t0))
				t0 += float(h)

			adam_bashforth(float(t0), float(h), int(quant), expfunc, order)

			for i in range(0, int(quant)+1):
				f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')			

			f_out.write('\n')

			plt.title("Metodo de Adams-Bashforth")

			show_graphic(vector_t, vector_y)

		elif inputf[0] == 'adam_multon':
			vector_y_am.clear()
			vector_t_am.clear()

			order = inputf[len(inputf)-1].split('\n')
			order = int(order[0])
			# print(order)
			for i in range(1, order):
				vector_y_am.append(float(inputf[i]))
			t0 = float(inputf[order])
			h = inputf[order+1]
			quant = inputf[order+2]
			expin = inputf[order+3].split('\n')
			expfunc = sympify(expin[0])

			f_out = open('saida.txt', 'a')
			f_out.write('Metodo de Adam-Multon' + '\n')
			f_out.write('y(' + str(t0) + ') = ' + str(vector_y_am[0]) + '\n')
			f_out.write('h = ' + h + '\n')
			
			# print(vector_y)
			for i in range(1, order+1):
				vector_t_am.append(float(t0))
				t0 += float(h)

			# print(quant)
			adam_multon(float(t0), float(h), int(quant), expfunc, order)

			for i in range(0, int(quant)+1):
				f_out.write(str(i) + ' ' + str(vector_y_am[i]) + '\n')

			f_out.write('\n')

			plt.title("Metodo de Adam-Multon")

			show_graphic(vector_t, vector_y)

		elif inputf[0] == 'formula_inversa':
			vector_y_fi.clear()
			vector_t_fi.clear()
			# vector_func_y.clear()

			order = inputf[len(inputf)-1].split('\n')
			order = int(order[0])
			# print(order)
			for i in range(1, order):
				vector_y_fi.append(float(inputf[i]))
			t0 = float(inputf[order])
			h = inputf[order+1]
			quant = inputf[order+2]
			expin = inputf[order+3].split('\n')
			expfunc = sympify(expin[0])

			f_out = open('saida.txt', 'a')
			f_out.write('Metodo Formula Inversa' + '\n')
			f_out.write('y(' + str(t0) + ') = ' + str(vector_y_fi[0]) + '\n')
			f_out.write('h = ' + h + '\n')

			# print(expfunc)
			for i in range(1, len(vector_y_fi)+1):
				vector_t_fi.append(float(t0))
				t0 += float(h)

			# print(vector_y_fi, vector_t_fi)
			formula_inversa(float(t0), float(h), int(quant), expfunc, order)

			for i in range(0, int(quant)+1):
				f_out.write(str(i) + ' ' + str(vector_y_fi[i]) + '\n')

			f_out.write('\n')

			plt.title("Metodo Formula Inversa")

			show_graphic(vector_t, vector_y)

		elif inputf[0] == 'adam_bashforth_by_euler' or inputf[0] == 'adam_bashforth_by_euler_inverso' or inputf[0] == 'adam_bashforth_by_euler_aprimorado' or inputf[0] == 'adam_bashforth_by_runge_kutta':
			y0 = inputf[1]
			t0 = inputf[2]
			h = inputf[3]
			quant = inputf[4]
			expin = inputf[5]
			expfunc = sympify(expin)
			order = inputf[6].split('\n')
			order = int(order[0])

			if inputf[0] == 'adam_bashforth_by_euler':
				vector_y.clear()
				vector_t.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Bashforth por Euler' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler(float(y0), float(t0), float(h), (order-1), expfunc)
				adam_bashforth(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adams-Bashforth por Euler")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'adam_bashforth_by_euler_inverso':
				vector_y.clear()
				vector_t.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Bashforth por Euler Inverso' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler_inverso(float(y0), float(t0), float(h), (order-1), expfunc)
				adam_bashforth(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adams-Bashforth por Euler Inverso")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'adam_bashforth_by_euler_aprimorado':
				vector_y.clear()
				vector_t.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Bashforth por Euler Aprimorado' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler_aprimorado(float(y0), float(t0), float(h), (order-1), expfunc)
				adam_bashforth(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adams-Bashforth por Euler Aprimorado")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'adam_bashforth_by_runge_kutta':
				vector_y.clear()
				vector_t.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Bashforth por Runge-Kutta' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = runge_kutta(float(y0), float(t0), float(h), (order-1), expfunc)
				adam_bashforth(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, int(quant)+1):
					f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adams-Bashforth por Runge-Kutta")

				show_graphic(vector_t, vector_y)

		elif inputf[0] == 'adam_multon_by_euler' or inputf[0] == 'adam_multon_by_euler_inverso' or inputf[0] == 'adam_multon_by_euler_aprimorado' or inputf[0] == 'adam_multon_by_runge_kutta':
			y0 = inputf[1]
			t0 = inputf[2]
			h = inputf[3]
			quant = inputf[4]
			expin = inputf[5]
			expfunc = sympify(expin)
			order = inputf[6].split('\n')
			order = int(order[0])

			if inputf[0] == 'adam_multon_by_euler':
				vector_y.clear()
				vector_t.clear()
				vector_y_am.clear()
				vector_t_am.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Multon por Euler' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_am.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_am.append(float(vector_t[i]))
					t0 += float(h)

				adam_multon(float(t0), float(h), int(quant), expfunc, order)

				# for i in range(0, 1)
				# 	f_out.write(str(i) + ' ' + str(vector_y[i]) + '\n')

				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_am[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adam-Multon por Euler")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'adam_multon_by_euler_inverso':
				vector_y.clear()
				vector_t.clear()
				vector_y_am.clear()
				vector_t_am.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Multon por Euler Inverso' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler_inverso(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_am.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_am.append(float(vector_t[i]))
					t0 += float(h)

				adam_multon(float(t0), float(h), int(quant), expfunc, order)

				# print(vector_y)

				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_am[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adam-Multon por Euler Inverso")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'adam_multon_by_euler_aprimorado':
				vector_y.clear()
				vector_t.clear()
				vector_y_am.clear()
				vector_t_am.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Multon por Euler Aprimorado' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler_aprimorado(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_am.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_am.append(float(vector_t[i]))
					t0 += float(h)

				adam_multon(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_am[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adam-Multon por Euler Aprimorado")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'adam_multon_by_runge_kutta':
				vector_y.clear()
				vector_t.clear()
				vector_y_am.clear()
				vector_t_am.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo de Adams-Multon por Runge-Kutta' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = runge_kutta(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_am.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_am.append(float(vector_t[i]))
					t0 += float(h)

				adam_multon(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_am[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo de Adam-Multon por Runge-Kutta")

				show_graphic(vector_t, vector_y)

		elif inputf[0] == 'formula_inversa_by_euler' or inputf[0] == 'formula_inversa_by_euler_inverso' or inputf[0] == 'formula_inversa_by_euler_aprimorado' or inputf[0] == 'formula_inversa_by_runge_kutta':
			y0 = inputf[1]
			t0 = inputf[2]
			h = inputf[3]
			quant = inputf[4]
			expin = inputf[5]
			expfunc = sympify(expin)
			order = inputf[6].split('\n')
			order = int(order[0])

			if inputf[0] == 'formula_inversa_by_euler':
				vector_y.clear()
				vector_t.clear()
				vector_y_fi.clear()
				vector_t_fi.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo Formula Inversa por Euler' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_fi.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_fi.append(float(vector_t[i]))
					t0 += float(h)

				formula_inversa(float(t0), float(h), int(quant), expfunc, order)
 
				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_fi[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo Formula Inversa por Euler")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'formula_inversa_by_euler_inverso':
				vector_y.clear()
				vector_t.clear()
				vector_y_fi.clear()
				vector_t_fi.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo Formula Inversa por Euler Inverso' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler_inverso(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_fi.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_fi.append(float(vector_t[i]))
					t0 += float(h)

				formula_inversa(float(t0), float(h), int(quant), expfunc, order)
 
				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_fi[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo Formula Inversa por Euler Inverso")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'formula_inversa_by_euler_aprimorado':
				vector_y.clear()
				vector_t.clear()
				vector_y_fi.clear()
				vector_t_fi.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo Formula Inversa por Euler Aprimorado' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = euler_aprimorado(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_fi.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_fi.append(float(vector_t[i]))
					t0 += float(h)

				formula_inversa(float(t0), float(h), int(quant), expfunc, order)
 
				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_fi[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo Formula Inversa por Euler Aprimorado")

				show_graphic(vector_t, vector_y)

			elif inputf[0] == 'formula_inversa_by_runge_kutta':
				vector_y.clear()
				vector_t.clear()
				vector_y_fi.clear()
				vector_t_fi.clear()

				f_out = open('saida.txt', 'a')
				f_out.write('Metodo Formula Inversa por Runge-Kutta' + '\n')
				f_out.write('y(' + str(t0) + ') = ' + y0 + '\n')
				f_out.write('h = ' + h + '\n')

				t0 = runge_kutta(float(y0), float(t0), float(h), (order-1), expfunc)
				
				for i in range(1, order-1):
					vector_y_fi.append(float(vector_y[i]))

				for i in range(1, order-1):
					vector_t_fi.append(float(vector_t[i]))
					t0 += float(h)

				formula_inversa(float(t0), float(h), int(quant), expfunc, order)

				for i in range(0, (int(quant)+1)):
					f_out.write(str(i) + ' ' + str(vector_y_fi[i]) + '\n')

				f_out.write('\n')

				plt.title("Metodo Formula Inversa por Runge-Kutta")

				show_graphic(vector_t, vector_y)

if __name__ == '__main__':
	main()