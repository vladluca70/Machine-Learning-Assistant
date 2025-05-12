from flask import Flask, render_template,request
import math
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def home():
    algoritmi=['LogisticRegression', 'ID3', 'AdaBoost', 'kNN_Algorithm', 'KMeans', 'NaiveBayes', 'OptimalBayes']
    return render_template("index.html", algoritmi=algoritmi)


def plot_points_with_red(points, x1, y1, elements):
    fig, ax = plt.subplots()
    positive_label_shown = False
    negative_label_shown = False


    ax.scatter(x1, y1, color='red', marker='o')

    for x, y, filled in points:
        index_in_elements = elements.index([x, y, filled])
        label = f'P{index_in_elements + 1}'

        if filled == 1:
            ax.scatter(x, y, color='black', marker='o')
            ax.text(x + 0.05, y + 0.05, label, color='black', fontsize=8)
            if not positive_label_shown:
                ax.scatter(x, y, color='black', marker='o', label='Punct clasificat pozitiv')
                positive_label_shown = True
        elif filled == -1:
            ax.scatter(x, y, color='black', marker='o', facecolors='none')
            ax.text(x + 0.05, y + 0.05, label, color='black', fontsize=8)
            if not negative_label_shown:
                ax.scatter(x, y, color='black', marker='o', facecolors='none', label='Punct clasificat negativ')
                negative_label_shown = True

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def knn_distance(x1,y1,x2,y2):
    dist=math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return round(dist,2)

def ordonare_puncte_knn(x1,x2,elements):
    lista_puncte=[]
    for i in range(len(elements)):
        linie=elements[i]
        distanta=knn_distance(x1,x2,linie[0], linie[1])
        lista_puncte.append([i,distanta])
    lista_sortata = sorted(lista_puncte, key=lambda x: x[1])
    return lista_sortata

@app.route("/kNN_Algorithm")
def kNN_Algorithm():
    return render_template("kNN_Algorithm.html")

@app.route("/kNN_AlgorithmExample", methods=["GET", "POST"])
def kNN_AlgorithmExample():
    elements = []
    n = None
    m = 3
    x1=None
    y1=None
    if request.method == "POST":
        n = int(request.form["n"]) if "n" in request.form else None
        x1=int(request.form["x1"]) if "x1" in request.form else None
        y1=int(request.form["y1"]) if "y1" in request.form else None

        if n:
            elements_form = request.form.getlist("elements[]")
            elements = [int(x) for x in elements_form]

    if isinstance(n, int):
        elements = [elements[i:i + m] for i in range(0, len(elements), m)]

    k_maxim=None
    k_lista=[]
    if n:
        if n%2==0:
            k_maxim=n-1
        else:
            k_maxim=n
        for i in range(1,n+1,2):
            k_lista.append(i)

    image = plot_points_with_red(elements,x1,y1,elements)

    ordonare_pentru_clasificare=ordonare_puncte_knn(x1,y1,elements)

    clasificare_knn=[]
    suma_clasificare=0
    if elements:
        for i in range(k_maxim):
            index_punct = ordonare_pentru_clasificare[i][0]
            clasa_punct = elements[index_punct][2]
            suma_clasificare += clasa_punct
            if i%2== 0:
                clasificare_knn.append(1 if suma_clasificare >= 0 else -1)

    imagini_clasificare=[]
    if elements:
        for i in range(0,k_maxim+1,2):
            puncte=[]
            for j in range(i+1):
                puncte.append(elements[ordonare_pentru_clasificare[j][0]])
            imagine=plot_points_with_red(puncte,x1,y1,elements)
            imagini_clasificare.append(imagine)

    if elements:
        print(clasificare_knn)
        print(ordonare_pentru_clasificare)

    return render_template("kNN_AlgorithmExample.html", n=n,x1=x1,y1=y1, elements=elements, image=image,
                           k_maxim=k_maxim, k_lista=k_lista,
                           imagini_clasificare=imagini_clasificare, clasificare_knn=clasificare_knn,
                           ordonare_pentru_clasificare=ordonare_pentru_clasificare)


@app.route("/LogisticRegression")
def LogisticRegression():
    return render_template("LogisticRegression.html")

@app.route("/LogisticRegressionExample", methods=["GET", "POST"])
def LogisticRegressionExample():
    elements = []
    n = m = None

    if request.method == "POST":
        n = int(request.form["n"]) if "n" in request.form else None
        m = int(request.form["m"]) if "m" in request.form else None

        if n and m:
            elements_form = request.form.getlist("elements[]")
            elements= [int(x) for sublist in elements_form for x in sublist]
    if isinstance(n,int) and isinstance(m,int):
        step = m+1
        result = []
        for i in range(0, len(elements), step):
            result.append(elements[i:i + step])
    if isinstance(n,int):
        elements=[elements[i:i+m+1] for i in range(0, len(elements), m+1)]

    explicatii_functie_log_verosimilitare=[]
    rez_final_log_ver=[]
    if elements:
        contor=1
        for linie in elements:
            rez_final_log_ver.append(linie)
            explicatii_pe_linie=[]
            explicatii_pe_linie.append(f"Linia {contor} din tabel:")
            if linie[m]==1:
                explicatii_pe_linie.append("Deoarece y=1, vom folosi prima parte a formulei")
                elemente_nenule_poz=[]
                for poz in range(m):
                    if linie[poz]!=0:
                        elemente_nenule_poz.append(poz+1)
                elemente_nenule_poz = ",".join(map(str, elemente_nenule_poz))
                explicatii_pe_linie.append(f"Vom folosi valorile variabilelor x de pe linia {contor} care sunt nenule."
                                           f"In cazul nostru, avem atributele x de pe coloanele {elemente_nenule_poz}")
                valori_cors_index=[]
                for kki in elemente_nenule_poz:
                    if kki!=",":
                        valori_cors_index.append(elements[contor-1][int(kki)-1])

                indici_pentru_calcul=[0]
                for ind in elemente_nenule_poz:
                    if ind!=",":
                        indici_pentru_calcul.append(int(ind))

                explicatii_pe_linie.append(indici_pentru_calcul)
                explicatii_pe_linie.append(str(contor))
                explicatii_pe_linie.append(valori_cors_index)
                explicatii_pe_linie.append(linie[m])
                explicatii_functie_log_verosimilitare.append(explicatii_pe_linie)
                contor+=1
            else:
                explicatii_pe_linie.append("Deoarece y=0, vom folosi a doua parte a formulei")
                elemente_nenule_poz = []
                for poz in range(m):
                    if linie[poz] != 0:
                        elemente_nenule_poz.append(poz + 1)
                elemente_nenule_poz = ",".join(map(str, elemente_nenule_poz))
                explicatii_pe_linie.append(f"Vom folosi valorile variabilelor x de pe linia {contor} care sunt nenule."
                                           f"In cazul nostru, avem atributele x de pe coloanele {elemente_nenule_poz}")
                valori_cors_index = []
                for kki in elemente_nenule_poz:
                    if kki != ",":
                        valori_cors_index.append(elements[contor - 1][int(kki) - 1])

                indici_pentru_calcul = [0]
                for ind in elemente_nenule_poz:
                    if ind != ",":
                        indici_pentru_calcul.append(int(ind))

                explicatii_pe_linie.append(indici_pentru_calcul)
                explicatii_pe_linie.append(str(contor))
                explicatii_pe_linie.append(valori_cors_index)
                explicatii_pe_linie.append(linie[m])
                explicatii_functie_log_verosimilitare.append(explicatii_pe_linie)
                contor += 1

    vector_gradient = []

    if elements:
        for linie in elements:
            expl = []
            vct = "(1"
            expl.append(linie[-1])
            for i in range(len(linie) - 1):
                vct = vct + "," + str(linie[i])
            vct = vct + ")ᵀ"
            expl.append(vct)
            vector_gradient.append(expl)

    valori_y=[]
    if elements:
        for i in range(n):
            linie=elements[i]
            valori_y.append(linie[m])

    x_coloane=[]
    if elements:
        prima_coloana_vector=[]
        for i in range(n):
            prima_coloana_vector.append(1)
        x_coloane.append(prima_coloana_vector)
    if elements:
        for i in range(m):
            coloana=[]
            for j in range(n):
                coloana.append(elements[j][i])
            x_coloane.append(coloana)

    linie_tabel_fara_y=[]
    if elements:
        for i in range(n):
            x=[]
            linie=elements[i]
            for j in linie:
                x.append(j)
            x.pop()
            linie_tabel_fara_y.append(x)

    return render_template("LogisticRegressionExample.html", n=n, m=m, elements=elements,
                           explicatii_functie_log_verosimilitare=explicatii_functie_log_verosimilitare,
                           rez_final_log_ver=rez_final_log_ver,vector_gradient=vector_gradient,
                           valori_y=valori_y,x_coloane=x_coloane,linie_tabel_fara_y=linie_tabel_fara_y)

@app.route("/ID3")
def ID3():
    return render_template("ID3.html")


@app.route("/ID3Example", methods=["GET", "POST"])
def ID3Example():
    n = None
    m = None
    header = []
    elements = []

    if request.method == 'POST':
        # Preluarea valorilor pentru n și m
        n = int(request.form.get('n', 0))
        m = int(request.form.get('m', 0))

        # Preluarea valorilor pentru header (prima linie)
        if 'header' in request.form:
            header = request.form.getlist('header[]')

        # Preluarea valorilor pentru tabel (restul elementelor)
        if 'elements' in request.form:
            elements = []
            for i in range(n):
                row = []
                for j in range(m):
                    value = int(request.form.get(f'elements[{i}][{j}]', 0))
                    row.append(value)
                elements.append(row)

        print("header:", header)
        print("elements:", elements)

    return render_template("ID3Example.html", n=n, m=m, header=header, elements=elements)



def plot_KMeans(first_table, second_table):
    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots()

    if first_table:
        x_first, y_first = zip(*first_table)
        ax.scatter(x_first, y_first, color='black', label='Puncte')

    if second_table:
        x_second, y_second = zip(*second_table)
        ax.scatter(x_second, y_second, color='red', marker='X', s=100, label='Centroizi')

    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return image_base64

def kmeans_distante(x1, x2, y1, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return round(dist, 2)

def calculeaza_nou_centroid(grup):
    if not grup:
        return (0, 0)
    x_mean = sum(x for x, _ in grup) / len(grup)
    y_mean = sum(y for _, y in grup) / len(grup)
    return (round(x_mean, 2), round(y_mean, 2))

def iteratii_kmeans(puncte, centroizi_initiali):
    toate_iteratiile = []

    centroizi = centroizi_initiali

    for _ in range(3):
        repartizare = {}
        for i in range(len(centroizi)):
            repartizare[i] = []

        for x1, y1 in puncte:
            distante = []
            for idx, (x2, y2) in enumerate(centroizi):
                dist = kmeans_distante(x1, x2, y1, y2)
                distante.append((dist, idx))

            _, idx_minim = min(distante)
            repartizare[idx_minim].append((x1, y1))

        noi_centroizi = []
        for idx in range(len(repartizare)):
            nou_centroid = calculeaza_nou_centroid(repartizare[idx])
            noi_centroizi.append(nou_centroid)

        toate_iteratiile.append((repartizare, noi_centroizi))

        centroizi = noi_centroizi

    return toate_iteratiile

def kmeans_grafice(repartizare, centroizi, it):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import numpy as np

    culori = ['green', 'red', 'yellow', 'purple', 'orange', 'blue', 'pink', 'cyan', 'brown', 'lime']
    fig, ax = plt.subplots()

    for idx, puncte in repartizare.items():
        if puncte:
            xs, ys = zip(*puncte)
            ax.scatter(xs, ys, c='black')

    for idx, (x, y) in enumerate(centroizi):
        ax.scatter(x, y, c=culori[idx % len(culori)], marker='X', s=200)

    for i in range(len(centroizi)):
        for j in range(i + 1, len(centroizi)):
            x1, y1 = centroizi[i]
            x2, y2 = centroizi[j]

            ax.plot([x1, x2], [y1, y2], linestyle='dotted', color='grey')

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            dx = x2 - x1
            dy = y2 - y1

            perp_dx = -dy
            perp_dy = dx

            lungime = np.sqrt(perp_dx**2 + perp_dy**2)
            if lungime == 0:
                continue
            perp_dx /= lungime
            perp_dy /= lungime

            offset = 5
            x_start = mid_x - perp_dx * offset
            x_end = mid_x + perp_dx * offset
            y_start = mid_y - perp_dy * offset
            y_end = mid_y + perp_dy * offset

            ax.plot([x_start, x_end], [y_start, y_end], color='blue')

    ax.legend()
    ax.set_title(f"Repartizare Puncte, Centroizi si Mediatoare la iteratia{it}")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    imagine_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return imagine_base64


@app.route("/KMeans")
def KMeans():
    return render_template("KMeans.html")

@app.route("/KMeansExample", methods=["GET", "POST"])
def KMeansExample():
    n = m = None
    first_table = second_table = None
    imagine_initiala = None

    kmeans_grafic_1=None
    kmeans_grafic_2=None
    kmeans_grafic_3=None

    repartizare_iteratia_1 = None
    repartizare_iteratia_2 = None
    repartizare_iteratia_3 = None

    centroizi_iteratia_1 = None
    centroizi_iteratia_2 = None
    centroizi_iteratia_3 = None

    calcul_suma_x_iteratia_1 = {}
    calcul_suma_y_iteratia_1 = {}
    calcul_suma_x_iteratia_2 = {}
    calcul_suma_y_iteratia_2 = {}
    calcul_suma_x_iteratia_3 = {}
    calcul_suma_y_iteratia_3 = {}

    if request.method == "POST":
        if "first_table[]" in request.form and "second_table[]" in request.form:
            n = int(request.form.get("n"))
            m = int(request.form.get("m"))
            first_data = request.form.getlist("first_table[]")
            second_data = request.form.getlist("second_table[]")

            first_table = []
            for i in range(0, len(first_data), 2):
                first_table.append((int(first_data[i]), int(first_data[i + 1])))

            second_table = []
            for i in range(0, len(second_data), 2):
                second_table.append((int(second_data[i]), int(second_data[i + 1])))

        else:
            n = int(request.form.get("n"))
            m = int(request.form.get("m"))

        if n and m:
            print("firsttable", first_table)
            print("secondtable", second_table)
            imagine_initiala = plot_KMeans(first_table, second_table)

        if first_table and second_table:
            rezultat = iteratii_kmeans(first_table, second_table)

            toate_repartizarile = []
            toti_centroizii = []

            for idx, (repartizare, centroizi) in enumerate(rezultat):
                toate_repartizarile.append(repartizare)
                toti_centroizii.append(centroizi)

                if idx == 0:
                    repartizare_iteratia_1 = repartizare
                    centroizi_iteratia_1 = centroizi
                elif idx == 1:
                    repartizare_iteratia_2 = repartizare
                    centroizi_iteratia_2 = centroizi
                elif idx == 2:
                    repartizare_iteratia_3 = repartizare
                    centroizi_iteratia_3 = centroizi

            # Calcul pentru repartizarea iteratiilor
            for cluster, points in repartizare_iteratia_1.items():
                sum_x = sum(point[0] for point in points)
                sum_y = sum(point[1] for point in points)
                calcul_suma_x_iteratia_1[cluster] = sum_x
                calcul_suma_y_iteratia_1[cluster] = sum_y

            for cluster, points in repartizare_iteratia_2.items():
                sum_x = sum(point[0] for point in points)
                sum_y = sum(point[1] for point in points)
                calcul_suma_x_iteratia_2[cluster] = sum_x
                calcul_suma_y_iteratia_2[cluster] = sum_y

            for cluster, points in repartizare_iteratia_3.items():
                sum_x = sum(point[0] for point in points)
                sum_y = sum(point[1] for point in points)
                calcul_suma_x_iteratia_3[cluster] = sum_x
                calcul_suma_y_iteratia_3[cluster] = sum_y

            # Calcul pentru centroizi (centroidul)
            for cluster in repartizare_iteratia_1:
                sum_x = calcul_suma_x_iteratia_1.get(cluster, 0)
                sum_y = calcul_suma_y_iteratia_1.get(cluster, 0)
                cardinal = len(repartizare_iteratia_1[cluster])
                if cardinal!=0:
                    centroizi_iteratia_1[cluster] = (sum_x / cardinal, sum_y / cardinal)

            for cluster in repartizare_iteratia_2:
                sum_x = calcul_suma_x_iteratia_2.get(cluster, 0)
                sum_y = calcul_suma_y_iteratia_2.get(cluster, 0)
                cardinal = len(repartizare_iteratia_2[cluster])
                if cardinal!=0:
                    centroizi_iteratia_2[cluster] = (sum_x / cardinal, sum_y / cardinal)

            for cluster in repartizare_iteratia_3:
                sum_x = calcul_suma_x_iteratia_3.get(cluster, 0)
                sum_y = calcul_suma_y_iteratia_3.get(cluster, 0)
                cardinal = len(repartizare_iteratia_3[cluster])
                if cardinal!=0:
                    centroizi_iteratia_3[cluster] = (sum_x / cardinal, sum_y / cardinal)

            if repartizare_iteratia_1 and centroizi_iteratia_1:
                kmeans_grafic_1 = kmeans_grafice(repartizare_iteratia_1, centroizi_iteratia_1, 1)

            if repartizare_iteratia_2 and centroizi_iteratia_2:
                kmeans_grafic_2 = kmeans_grafice(repartizare_iteratia_2, centroizi_iteratia_2, 2)

            if repartizare_iteratia_3 and centroizi_iteratia_3:
                kmeans_grafic_3 = kmeans_grafice(repartizare_iteratia_3, centroizi_iteratia_3, 3)

    return render_template(
        "KMeansExample.html",
        n=n, m=m,
        first_table=first_table,
        second_table=second_table,
        imagine_initiala=imagine_initiala,
        repartizare_iteratia_1=repartizare_iteratia_1,
        repartizare_iteratia_2=repartizare_iteratia_2,
        repartizare_iteratia_3=repartizare_iteratia_3,
        centroizi_iteratia_1=centroizi_iteratia_1,
        centroizi_iteratia_2=centroizi_iteratia_2,
        centroizi_iteratia_3=centroizi_iteratia_3,
        kmeans_grafic_1=kmeans_grafic_1,
        kmeans_grafic_2=kmeans_grafic_2,
        kmeans_grafic_3=kmeans_grafic_3,
        calcul_suma_x_iteratia_1=calcul_suma_x_iteratia_1,
        calcul_suma_y_iteratia_1=calcul_suma_y_iteratia_1,
        calcul_suma_x_iteratia_2=calcul_suma_x_iteratia_2,
        calcul_suma_y_iteratia_2=calcul_suma_y_iteratia_2,
        calcul_suma_x_iteratia_3=calcul_suma_x_iteratia_3,
        calcul_suma_y_iteratia_3=calcul_suma_y_iteratia_3
    )


from collections import Counter, defaultdict


def compute_naive_bayes_probabilities(values, prediction):
    n = len(values)
    m = len(prediction)

    values_by_class = defaultdict(list)
    for row in values:
        *features, label = row
        values_by_class[label].append(features)

    total_samples = sum(len(v) for v in values_by_class.values())
    p_k = {k: len(v) / total_samples for k, v in values_by_class.items()}

    conditional_probs = {}
    probabilities_p0 = []
    probabilities_p1 = []

    for k in [0, 1]:
        rows = values_by_class.get(k, [])
        total_rows = len(rows)

        probs = []
        for j in range(m):
            column_values = [row[j] for row in rows]
            feature_counts = Counter(column_values)
            count = feature_counts.get(prediction[j], 0)

            prob = count / total_rows
            probs.append(prob)

            if k == 0:
                probabilities_p0.append(prob)
            else:
                probabilities_p1.append(prob)

        conditional_probs[k] = probs

    def product(lst):
        result = 1.0
        for x in lst:
            result *= x
        return result

    p0 = product(probabilities_p0) * p_k.get(0, 0)
    p1 = product(probabilities_p1) * p_k.get(1, 0)

    return p0, p1, probabilities_p0, probabilities_p1

def calculate_k_proportions(values):
    total_samples = len(values)
    k0_count = sum(1 for row in values if row[-1] == 0)
    k1_count = total_samples - k0_count

    k0_proportion = k0_count / total_samples
    k1_proportion = k1_count / total_samples

    return k0_proportion, k1_proportion

@app.route("/NaiveBayes")
def NaiveBayes():
    return render_template("NaiveBayes.html")

@app.route("/NaiveBayesExample", methods=["GET", "POST"])
def NaiveBayesExample():
    if request.method == "POST":
        n = request.form.get("n")
        m = request.form.get("m")

        if n and m and "values[]" not in request.form:
            n = int(n)
            m = int(m)
            return render_template("NaiveBayesExample.html", n=n, m=m)

        elif "values[]" in request.form:
            n = int(request.form.get("n"))
            m = int(request.form.get("m"))

            values_flat = request.form.getlist("values[]")
            values = [
                [
                    int(values_flat[i * (m + 1) + j]) if j < m else int(values_flat[i * (m + 1) + j])
                    for j in range(m + 1)
                ]
                for i in range(n)
            ]

            prediction_flat = request.form.getlist("prediction[]")
            prediction = [int(x) for x in prediction_flat]

            print("Tabel:", values)
            print("Exemplu de predicție:", prediction)

            p0, p1, probabilities_p0, probabilities_p1 = compute_naive_bayes_probabilities(values, prediction)
            print(f"p0 = {p0}")
            print(f"p1 = {p1}")
            print("\nProbabilitățile individuale pentru p0:", probabilities_p0)
            print("Probabilitățile individuale pentru p1:", probabilities_p1)
            k0, k1 = calculate_k_proportions(values)
            print(k0)
            print(k1)
            p0_gt_p1=p0/(p0+p1)
            p1_gt_p0=p1/(p0+p1)
            return render_template("NaiveBayesExample.html", n=n, m=m, values=values, prediction=prediction,
                                   p0=p0, p1=p1, probabilities_p0=probabilities_p0, probabilities_p1=probabilities_p1,
                                   k0=k0, k1=k1, p0_gt_p1=p0_gt_p1, p1_gt_p0=p1_gt_p0)

    return render_template("NaiveBayesExample.html")


@app.route("/OptimalBayes")
def OptimalBayes():
    return render_template("OptimalBayes.html")


def count_prediction_and_k(table, prediction):
    x0 = 0
    x1 = 0
    k0 = 0
    k1 = 0
    total_rows = len(table)

    for row in table:
        if row[:-1] == prediction:
            if row[-1] == 0:
                x0 += 1
            elif row[-1] == 1:
                x1 += 1
        if row[-1] == 0:
            k0 += 1
        elif row[-1] == 1:
            k1 += 1

    k0_ratio = k0 / total_rows if total_rows > 0 else 0
    k1_ratio = k1 / total_rows if total_rows > 0 else 0

    return x0, x1, k0_ratio, k1_ratio

@app.route("/OptimalBayesExample", methods=["GET", "POST"])
def OptimalBayesExample():
    if request.method == "POST":
        n = request.form.get("n")
        m = request.form.get("m")

        if n and m and "values[]" not in request.form:
            n = int(n)
            m = int(m)
            return render_template("OptimalBayesExample.html", n=n, m=m)

        elif "values[]" in request.form:
            n = int(request.form.get("n"))
            m = int(request.form.get("m"))

            values_flat = request.form.getlist("values[]")
            values = [
                [
                    int(values_flat[i * (m + 1) + j]) if j < m else int(values_flat[i * (m + 1) + j])
                    for j in range(m + 1)
                ]
                for i in range(n)
            ]

            prediction_flat = request.form.getlist("prediction[]")
            prediction = [int(x) for x in prediction_flat]

            print("Tabel:", values)
            print("Exemplu de predicție:", prediction)

            x0,x1,k0,k1=count_prediction_and_k(values, prediction)
            p0=x0*k0
            p1=x1*k1
            p0_gt_p1=p0/(p0+p1)
            p1_gt_p0=p1/(p1+p0)
            print("x")
            print(x0)
            print(x1)
            return render_template("OptimalBayesExample.html", n=n, m=m, values=values, prediction=prediction,
                                   p0=p0, p1=p1,
                                   k0=k0, k1=k1, p0_gt_p1=p0_gt_p1, p1_gt_p0=p1_gt_p0, x0=x0, x1=x1)

    return render_template("OptimalBayesExample.html")


def adaboost_plot1(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Listele trebuie să aibă aceeași lungime")

    fig, ax = plt.subplots()

    for (x, y), label in zip(list1, list2):
        if label == 1:
            ax.plot(x, y, 'ko')
        elif label == -1:
            ax.plot(x, y, 'ko', markerfacecolor='white')
        else:
            raise ValueError("Etichetele trebuie să fie 1 sau -1")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Reprezentare puncte cu etichete')
    ax.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64


def praguri(coords, labels, axis):
    from collections import defaultdict
    sorted_data = sorted(zip(coords, labels), key=lambda x: x[0][axis])

    grouped = defaultdict(list)
    for point, label in sorted_data:
        grouped[point[axis]].append(label)
    sorted_values = sorted(grouped.keys())

    thresholds = []
    for i in range(len(sorted_values) - 1):
        v1, v2 = sorted_values[i], sorted_values[i+1]
        labels1, labels2 = grouped[v1], grouped[v2]
        if any(l1 != l2 for l1 in labels1 for l2 in labels2):
            thresholds.append((v1 + v2) / 2)
    return thresholds

def praguri_adaboost(coords, labels):
    thresholds_x = praguri(coords, labels, axis=0)
    thresholds_y = praguri(coords, labels, axis=1)
    return thresholds_x, thresholds_y

def calculare_erori_axa_x_adaboost(coordonate, etichete, pragurix, ponderi):
    erori_stanga = []
    erori_dreapta = []

    for prag in pragurix:
        eroare_stanga = 0
        eroare_dreapta = 0
        for i, (x, _) in enumerate(coordonate):
            eticheta_real = etichete[i]
            pondere = ponderi[i]

            eticheta_pred_stanga = 1 if x < prag else -1
            if eticheta_pred_stanga != eticheta_real:
                eroare_stanga += pondere

            eticheta_pred_dreapta = 1 if x >= prag else -1
            if eticheta_pred_dreapta != eticheta_real:
                eroare_dreapta += pondere

        erori_stanga.append(eroare_stanga)
        erori_dreapta.append(eroare_dreapta)

    return erori_stanga, erori_dreapta

def calculare_erori_axa_y_adaboost(coordonate, etichete, praguriy, ponderi):
    erori_jos = []
    erori_sus = []

    for prag in praguriy:
        eroare_jos = 0
        eroare_sus = 0
        for i in range(len(coordonate)):
            y = coordonate[i][1]
            y_real = etichete[i]
            w = ponderi[i]

            y_pred_jos = 1 if y < prag else -1
            if y_pred_jos != y_real:
                eroare_jos += w

            y_pred_sus = 1 if y >= prag else -1
            if y_pred_sus != y_real:
                eroare_sus += w

        erori_jos.append(eroare_jos)
        erori_sus.append(eroare_sus)

    return erori_jos, erori_sus

def grafic_adaboost_cu_praguri(puncte, etichete, pragurix, praguriy):
    from io import BytesIO
    fig, ax = plt.subplots()
    for (x, y), et in zip(puncte, etichete):
        if et == 1:
            ax.plot(x, y, 'ko')  # 'k' = negru, 'o' = punct
        else:
            ax.plot(x, y, 'wo', markeredgecolor='black')  # punct gol cu contur negru

    for x in pragurix:
        ax.axvline(x=x, color='green', linestyle='--')
    for y in praguriy:
        ax.axhline(y=y, color='red', linestyle='--')
    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', alpha=0.5)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)

    return img_base64

def verifica_clasificare(puncte, etichete, mesaj):
    corect = []
    gresit = []
    idx_corect = []
    idx_gresit = []

    # extrage coordonata, operatorul si pragul
    import re
    match = re.match(r"(x|y)(<|>=)([\d.]+)", mesaj)
    if not match:
        raise ValueError("Mesaj invalid. Format acceptat: 'x<1.5', 'y>=3.0', etc.")

    coord, op, prag = match.groups()
    prag = float(prag)
    index = 0 if coord == "x" else 1

    # clasificare și verificare
    for i, (punct, et) in enumerate(zip(puncte, etichete)):
        val = punct[index]
        if op == "<":
            pred = 1 if val < prag else -1
        else:  # op == ">="
            pred = 1 if val >= prag else -1

        if pred == et:
            corect.append(tuple(punct))
            idx_corect.append(i)
        else:
            gresit.append(tuple(punct))
            idx_gresit.append(i)

    return corect, gresit, idx_gresit, idx_corect


def adaboost_desen_final(puncte, etichete, mesaj1, mesaj2, mesaj3):
    def interpreteaza_mesaj(mesaj):
        axa = 'y' if mesaj[0] == 'y' else 'x'
        if '<=' in mesaj:
            valoare = float(mesaj.split('<=')[1])
            sens = '<='
        elif '>=' in mesaj:
            valoare = float(mesaj.split('>=')[1])
            sens = '>='
        elif '<' in mesaj:
            valoare = float(mesaj.split('<')[1])
            sens = '<'
        elif '>' in mesaj:
            valoare = float(mesaj.split('>')[1])
            sens = '>'
        return axa, sens, valoare

    linii = [mesaj1, mesaj2, mesaj3]
    instructiuni = [interpreteaza_mesaj(m) for m in linii]
    culori = ['red', 'green', 'blue']

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for (x, y), et in zip(puncte, etichete):
        if et == 1:
            ax.plot(x, y, 'ko')  # cerc negru
        else:
            ax.plot(x, y, 'wo', markeredgecolor='k')  # cerc gol
    ax.set_xlim(0, max(x for x, y in puncte) + 2)
    ax.set_ylim(0, max(y for x, y in puncte) + 2)

    for (axa, semn, val), culoare in zip(instructiuni, culori):
        if axa == 'x':
            ax.axvline(x=val, color=culoare, linestyle='--')
            y_min, y_max = ax.get_ylim()
            y_pos = y_max - 0.3
            if semn in ['<', '<=']:
                ax.text(val - 0.1, y_pos, '+', ha='right', va='center', fontsize=12, color=culoare)
                ax.text(val + 0.1, y_pos, '-', ha='left', va='center', fontsize=12, color=culoare)
            else:
                ax.text(val - 0.1, y_pos, '-', ha='right', va='center', fontsize=12, color=culoare)
                ax.text(val + 0.1, y_pos, '+', ha='left', va='center', fontsize=12, color=culoare)
        else:
            ax.axhline(y=val, color=culoare, linestyle='--')
            x_min, x_max = ax.get_xlim()
            x_pos = x_max - 0.3
            if semn in ['<', '<=']:
                ax.text(x_pos, val + 0.2, '-', ha='center', va='bottom', fontsize=12, color=culoare)
                ax.text(x_pos, val - 0.2, '+', ha='center', va='top', fontsize=12, color=culoare)
            else:
                ax.text(x_pos, val + 0.2, '+', ha='center', va='bottom', fontsize=12, color=culoare)
                ax.text(x_pos, val - 0.2, '-', ha='center', va='top', fontsize=12, color=culoare)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    imagine_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return imagine_base64


@app.route("/AdaBoost")
def AdaBoost():
    return render_template("AdaBoost.html")

@app.route("/AdaBoostExample", methods=["GET", "POST"])
def AdaBoostExample():
    elements = []
    n = None

    if request.method == "POST":
        n = int(request.form["n"]) if "n" in request.form else None

        if n:
            elements_form = request.form.getlist("elements[]")
            try:
                elements = [int(x) for x in elements_form]

                if len(elements) == 3 * n:
                    elements = [elements[i:i + 3] for i in range(0, len(elements), 3)]
                else:
                    elements = []
            except ValueError:
                elements = []
    etichete=[]
    coordonate=[]
    imagine1=[]
    imagine_cu_praguri=[]
    praguriX=[]
    praguriY=[]
    prag_exterior_x=None
    prag_exterior_y=None
    nr_puncte=None
    ponderi_it1=[]
    it1_axa_x_erori1=[]
    it1_axa_x_erori2=[]
    it1_axa_y_erori1=[]
    it1_axa_y_erori2=[]
    it1_eroare_minima_tabel=9999999
    it1_prag_pentru_noi_ponderi=""
    prag_pentru_eroare_minima_it1_valoare=None
    prag_pentru_eroare_minima_it1_mesaj=""
    it1_puncte_clasificate_corect=[]
    it1_puncte_clasificate_gresit=[]
    it1_index_puncte_clasificate_corect=[]
    it1_index_puncte_clasificate_gresit=[]
    ponderi_it2=[]
    gamma1=None
    alpha1=None
    valoare_X_it1_prag=None

    it2_axa_x_erori1=[]
    it2_axa_x_erori2=[]
    it2_axa_y_erori1=[]
    it2_axa_y_erori2=[]
    it2_eroare_minima_tabel=9999999
    gamma2=None
    alpha2=None
    it2_prag_pentru_noi_ponderi=""
    prag_pentru_eroare_minima_it2_valoare=None
    prag_pentru_eroare_minima_it2_mesaj=""
    valoare_X_it2_prag=None
    it2_puncte_clasificate_corect=[]
    it2_puncte_clasificate_gresit=[]
    it2_index_puncte_clasificate_corect=[]
    it2_index_puncte_clasificate_gresit=[]
    ponderi_it3=[]

    it3_axa_x_erori1=[]
    it3_axa_x_erori2=[]
    it3_axa_y_erori1=[]
    it3_axa_y_erori2=[]
    it3_eroare_minima_tabel=9999999
    gamma3=None
    alpha3=None
    it3_prag_pentru_noi_ponderi=""
    prag_pentru_eroare_minima_it3_valoare=None
    prag_pentru_eroare_minima_it3_mesaj=""
    valoare_X_it3_prag=None
    it3_puncte_clasificate_corect=[]
    it3_puncte_clasificate_gresit=[]
    it3_index_puncte_clasificate_corect=[]
    it3_index_puncte_clasificate_gresit=[]
    ponderi_it4=[]
    desen_final=[]
    if elements:
        for linie in elements:
            etichete.append(linie[2])
            coordonate.append([linie[0],linie[1]])
    if elements:
        for i in range(len(etichete)):
            ponderi_it1.append(round((1 / len(etichete)),4))

        print("coordonate", coordonate)
        print("etichete", etichete)
        imagine1=adaboost_plot1(coordonate,etichete)
        praguriX,praguriY=praguri_adaboost(coordonate, etichete)
        print("pragurix",praguriX)
        print("praguriY", praguriY)
        prag_exterior_x = min(p[0] for p in coordonate)-0.5
        prag_exterior_y = min(p[1] for p in coordonate)-0.5

        praguriX.insert(0, prag_exterior_x)
        praguriY.insert(0, prag_exterior_y )
        nr_puncte=len(coordonate)
        imagine_cu_praguri=grafic_adaboost_cu_praguri(coordonate, etichete, praguriX, praguriY)
        #iteratie 1
        it1_axa_x_erori1, it1_axa_x_erori2=calculare_erori_axa_x_adaboost(coordonate,etichete, praguriX, ponderi_it1)
        it1_axa_y_erori1, it1_axa_y_erori2=calculare_erori_axa_y_adaboost(coordonate, etichete, praguriY, ponderi_it1)
        it1_eroare_minima_tabel=min(it1_axa_x_erori1+it1_axa_x_erori2+it1_axa_y_erori1+it1_axa_y_erori2)

        ok=1
        for i in range(len(it1_axa_x_erori1)):
            if it1_axa_x_erori1[i]==it1_eroare_minima_tabel:
                if ok==1:
                    it1_prag_pentru_noi_ponderi+=f"x<{praguriX[i]}"
                    prag_pentru_eroare_minima_it1_mesaj += f"X\u2081 < {praguriX[i]}"
                    prag_pentru_eroare_minima_it1_valoare=praguriX[i]
                    valoare_X_it1_prag=1
                    ok=0
        for i in range(len(it1_axa_x_erori2)):
            if it1_axa_x_erori2[i]==it1_eroare_minima_tabel:
                if ok==1:
                    it1_prag_pentru_noi_ponderi += f"x>={praguriX[i]}"
                    prag_pentru_eroare_minima_it1_mesaj+=f"X\u2081 >={praguriX[i]}"
                    prag_pentru_eroare_minima_it1_valoare=praguriX[i]
                    valoare_X_it1_prag=2
                    ok=0
        for i in range(len(it1_axa_y_erori1)):
            if it1_axa_y_erori1[i]==it1_eroare_minima_tabel:
                if ok==1:
                    it1_prag_pentru_noi_ponderi += f"y<{praguriY[i]}"
                    prag_pentru_eroare_minima_it1_mesaj+=f"X\u2082 <{praguriY[i]}"
                    prag_pentru_eroare_minima_it1_valoare=praguriY[i]
                    valoare_X_it1_prag=3
                    ok=0
        for i in range(len(it1_axa_y_erori2)):
            if it1_axa_y_erori2[i]==it1_eroare_minima_tabel:
                if ok==1:
                    it1_prag_pentru_noi_ponderi += f"y>={praguriY[i]}"
                    prag_pentru_eroare_minima_it1_mesaj+=f"X\u2082 >={praguriY[i]}"
                    prag_pentru_eroare_minima_it1_valoare=praguriY[i]
                    valoare_X_it1_prag=4
                    ok=0
        gamma1=0.5-it1_eroare_minima_tabel
        alpha1=0.5*(math.log((1-it1_eroare_minima_tabel)/it1_eroare_minima_tabel))
        it1_puncte_clasificate_corect,it1_puncte_clasificate_gresit, it1_index_puncte_clasificate_corect, it1_index_puncte_clasificate_gresit=verifica_clasificare(coordonate, etichete, it1_prag_pentru_noi_ponderi)

        #iteratia=2
        for i in ponderi_it1:
            ponderi_it2.append(i)
        for i in it1_index_puncte_clasificate_corect:
            ponderi_it2[i]=0.5/(len(it1_index_puncte_clasificate_corect)/len(etichete))*ponderi_it2[i]
        for i in it1_index_puncte_clasificate_gresit:
            ponderi_it2[i]=0.5/(len(it1_index_puncte_clasificate_gresit)/len(etichete))*ponderi_it2[i]

        it2_axa_x_erori1, it2_axa_x_erori2 = calculare_erori_axa_x_adaboost(coordonate, etichete, praguriX, ponderi_it2)
        it2_axa_y_erori1, it2_axa_y_erori2 = calculare_erori_axa_y_adaboost(coordonate, etichete, praguriY, ponderi_it2)
        it2_eroare_minima_tabel = min(it2_axa_x_erori1 + it2_axa_x_erori2 + it2_axa_y_erori1 + it2_axa_y_erori2)

        gamma2=0.5-it2_eroare_minima_tabel
        alpha2=0.5*(math.log((1-it2_eroare_minima_tabel)/it2_eroare_minima_tabel))

        ok2=1
        for i in range(len(it2_axa_x_erori1)):
            if it2_axa_x_erori1[i]==it2_eroare_minima_tabel:
                if ok2==1:
                    it2_prag_pentru_noi_ponderi+=f"x<{praguriX[i]}"
                    prag_pentru_eroare_minima_it2_mesaj += f"X\u2081 < {praguriX[i]}"
                    prag_pentru_eroare_minima_it2_valoare=praguriX[i]
                    valoare_X_it2_prag=1
                    ok2=0
        for i in range(len(it2_axa_x_erori2)):
            if it2_axa_x_erori2[i]==it2_eroare_minima_tabel:
                if ok2==1:
                    it2_prag_pentru_noi_ponderi += f"x>={praguriX[i]}"
                    prag_pentru_eroare_minima_it2_mesaj+=f"X\u2081 >={praguriX[i]}"
                    prag_pentru_eroare_minima_it2_valoare=praguriX[i]
                    valoare_X_it2_prag=2
                    ok2=0
        for i in range(len(it2_axa_y_erori1)):
            if it2_axa_y_erori1[i]==it2_eroare_minima_tabel:
                if ok2==1:
                    it2_prag_pentru_noi_ponderi += f"y<{praguriY[i]}"
                    prag_pentru_eroare_minima_it2_mesaj+=f"X\u2082 <{praguriY[i]}"
                    prag_pentru_eroare_minima_it2_valoare=praguriY[i]
                    valoare_X_it2_prag=3
                    ok2=0
        for i in range(len(it2_axa_y_erori2)):
            if it2_axa_y_erori2[i]==it2_eroare_minima_tabel:
                if ok2==1:
                    it2_prag_pentru_noi_ponderi += f"y>={praguriY[i]}"
                    prag_pentru_eroare_minima_it2_mesaj+=f"X\u2082 >={praguriY[i]}"
                    prag_pentru_eroare_minima_it2_valoare=praguriY[i]
                    valoare_X_it2_prag=4
                    ok2=0
        it2_puncte_clasificate_corect,it2_puncte_clasificate_gresit, it2_index_puncte_clasificate_corect, it2_index_puncte_clasificate_gresit=verifica_clasificare(coordonate, etichete, it2_prag_pentru_noi_ponderi)
        for i in ponderi_it2:
            ponderi_it3.append(i)
        for i in it2_index_puncte_clasificate_corect:
            ponderi_it3[i]=0.5/(len(it2_index_puncte_clasificate_corect)/len(etichete))*ponderi_it3[i]
        for i in it1_index_puncte_clasificate_gresit:
            ponderi_it3[i]=0.5/(len(it2_index_puncte_clasificate_gresit)/len(etichete))*ponderi_it3[i]

        # iteratia=3

        it3_axa_x_erori1, it3_axa_x_erori2 = calculare_erori_axa_x_adaboost(coordonate, etichete, praguriX,
                                                                            ponderi_it3)
        it3_axa_y_erori1, it3_axa_y_erori2 = calculare_erori_axa_y_adaboost(coordonate, etichete, praguriY,
                                                                            ponderi_it3)
        it3_eroare_minima_tabel = min(it3_axa_x_erori1 + it3_axa_x_erori2 + it3_axa_y_erori1 + it3_axa_y_erori2)

        gamma3 = 0.5 - it3_eroare_minima_tabel
        alpha3 = 0.5 * (math.log((1 - it3_eroare_minima_tabel) / it3_eroare_minima_tabel))

        ok3 = 1
        for i in range(len(it3_axa_x_erori1)):
            if it3_axa_x_erori1[i] == it3_eroare_minima_tabel:
                if ok3== 1:
                    it3_prag_pentru_noi_ponderi += f"x<{praguriX[i]}"
                    prag_pentru_eroare_minima_it3_mesaj += f"X\u2081 < {praguriX[i]}"
                    prag_pentru_eroare_minima_it3_valoare = praguriX[i]
                    valoare_X_it3_prag = 1
                    ok3 = 0
        for i in range(len(it3_axa_x_erori2)):
            if it3_axa_x_erori2[i] == it3_eroare_minima_tabel:
                if ok3== 1:
                    it3_prag_pentru_noi_ponderi += f"x>={praguriX[i]}"
                    prag_pentru_eroare_minima_it3_mesaj += f"X\u2081 >={praguriX[i]}"
                    prag_pentru_eroare_minima_it3_valoare = praguriX[i]
                    valoare_X_it3_prag = 2
                    ok3= 0
        for i in range(len(it3_axa_y_erori1)):
            if it3_axa_y_erori1[i] == it3_eroare_minima_tabel:
                if ok3 == 1:
                    it3_prag_pentru_noi_ponderi += f"y<{praguriY[i]}"
                    prag_pentru_eroare_minima_it3_mesaj += f"X\u2082 <{praguriY[i]}"
                    prag_pentru_eroare_minima_it3_valoare = praguriY[i]
                    valoare_X_it3_prag = 3
                    ok3= 0
        for i in range(len(it3_axa_y_erori2)):
            if it3_axa_y_erori2[i] == it3_eroare_minima_tabel:
                if ok3== 1:
                    it3_prag_pentru_noi_ponderi += f"y>={praguriY[i]}"
                    prag_pentru_eroare_minima_it3_mesaj += f"X\u2082 >={praguriY[i]}"
                    prag_pentru_eroare_minima_it3_valoare = praguriY[i]
                    valoare_X_it3_prag = 4
                    ok3= 0
        it3_puncte_clasificate_corect, it3_puncte_clasificate_gresit, it3_index_puncte_clasificate_corect, it3_index_puncte_clasificate_gresit = verifica_clasificare(
            coordonate, etichete, it3_prag_pentru_noi_ponderi)
        for i in ponderi_it3:
            ponderi_it4.append(i)
        for i in it3_index_puncte_clasificate_corect:
            ponderi_it4[i] = 0.5 / (len(it3_index_puncte_clasificate_corect) / len(etichete)) * ponderi_it4[i]
        for i in it1_index_puncte_clasificate_gresit:
            ponderi_it4[i] = 0.5 / (len(it3_index_puncte_clasificate_gresit) / len(etichete)) * ponderi_it4[i]

        desen_final=adaboost_desen_final(coordonate, etichete, it1_prag_pentru_noi_ponderi, it2_prag_pentru_noi_ponderi, it3_prag_pentru_noi_ponderi)
    return render_template("AdaBoostExample.html", n=n, elements=elements,
                           coordonate=coordonate, etichete=etichete, imagine1=imagine1,
                           praguriX=praguriX, praguriY=praguriY,
                           prag_exterior_x=prag_exterior_x, prag_exterior_y=prag_exterior_y,
                           nr_puncte=nr_puncte,imagine_cu_praguri=imagine_cu_praguri,
                           ponderi_it1=ponderi_it1, it1_axa_x_erori1=it1_axa_x_erori1,it1_axa_x_erori2=it1_axa_x_erori2,
                           it1_axa_y_erori1=it1_axa_y_erori1, it1_axa_y_erori2=it1_axa_y_erori2, it1_eroare_minima_tabel=it1_eroare_minima_tabel,
                           prag_pentru_eroare_minima_it1_mesaj=prag_pentru_eroare_minima_it1_mesaj, prag_pentru_eroare_minima_it1_valoare=prag_pentru_eroare_minima_it1_valoare,
                           gamma1=gamma1, alpha1=alpha1, valoare_X_it1_prag=valoare_X_it1_prag,
                           it1_puncte_clasificate_corect=it1_puncte_clasificate_corect,
                           it1_puncte_clasificate_gresit=it1_puncte_clasificate_gresit,
                           it1_index_puncte_clasificate_corect=it1_index_puncte_clasificate_corect,
                           it1_index_puncte_clasificate_gresit=it1_index_puncte_clasificate_gresit,
                           ponderi_it2=ponderi_it2,it2_eroare_minima_tabel=it2_eroare_minima_tabel,
                           it2_axa_x_erori1=it2_axa_x_erori1,it2_axa_x_erori2=it2_axa_x_erori2,
                           it2_axa_y_erori1=it2_axa_y_erori1,it2_axa_y_erori2=it2_axa_y_erori2,
                           alpha2=alpha2, gamma2=gamma2,
                           it2_prag_pentru_noi_ponderi=it2_prag_pentru_noi_ponderi,
                           prag_pentru_eroare_minima_it2_mesaj=prag_pentru_eroare_minima_it2_mesaj,
                           prag_pentru_eroare_minima_it2_valoare=prag_pentru_eroare_minima_it2_valoare,
                           valoare_X_it2_prag=valoare_X_it2_prag,
                           it2_puncte_clasificate_corect=it2_puncte_clasificate_corect,
                           it2_puncte_clasificate_gresit=it2_puncte_clasificate_gresit,
                           it2_index_puncte_clasificate_corect=it2_index_puncte_clasificate_corect,
                           it2_index_puncte_clasificate_gresit=it2_index_puncte_clasificate_gresit,
                           ponderi_it3=ponderi_it3,it3_eroare_minima_tabel=it3_eroare_minima_tabel,
                           it3_axa_x_erori1=it3_axa_x_erori1,it3_axa_x_erori2=it3_axa_x_erori2,
                           it3_axa_y_erori1=it3_axa_y_erori1,it3_axa_y_erori2=it3_axa_y_erori2,
                           alpha3=alpha3, gamma3=gamma3,
                           it3_prag_pentru_noi_ponderi=it3_prag_pentru_noi_ponderi,
                           prag_pentru_eroare_minima_it3_mesaj=prag_pentru_eroare_minima_it3_mesaj,
                           prag_pentru_eroare_minima_it3_valoare=prag_pentru_eroare_minima_it3_valoare,
                           valoare_X_it3_prag=valoare_X_it3_prag,
                           it3_puncte_clasificate_corect=it3_puncte_clasificate_corect,
                           it3_puncte_clasificate_gresit=it3_puncte_clasificate_gresit,
                           it3_index_puncte_clasificate_corect=it3_index_puncte_clasificate_corect,
                           it3_index_puncte_clasificate_gresit=it3_index_puncte_clasificate_gresit,
                           ponderi_it4=ponderi_it4, desen_final=desen_final
                           )

if __name__ == "__main__":
    app.run(debug=True)