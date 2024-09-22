#include <stdlib.h>
#include <unistd.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits>
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

#define EXPECTED_CLUSTERS 16
#define MIN_CLUSTER_SIZE 20
#define MAX_CLUSTER_SIZE 200
#define TEAM_SIZE 3
#define FIELD_SIZE_X 1440
#define FIELD_SIZE_Y 1240


typedef struct{
  int size;
  int xmin;
  int xmax;
  int ymin;
  int ymax;
  float centerx;
  float centery;
  float diffx;
  float diffy;
}cluster;

typedef struct{
  int i;
  int j;
  float min;
}relation;

typedef struct{
  int id;
  int uni_i;
  int uni_j;
  int red_k;
  float min_ij;
  float angle;
  float centerx;
  float centery;
}Robot;

typedef struct{
  int size;
  Robot robot[TEAM_SIZE];
}team;

typedef struct{
  int len;
  cluster* clus;
}cluster_vec;

typedef struct{
  float lowR;
  float highR;

  float lowG;
  float highG;

  float lowB;
  float highB;
}colors;

float EuclideanDistance(float xa, float ya, float xb, float yb){
  return sqrt((xa-xb)*(xa-xb)+(ya-yb)*(ya-yb));
}

void zeraTime(team* time) {
	time->size = 0;
	for (int i = 0; i < TEAM_SIZE; i++) {
		time->robot[i].id = -1;
		time->robot[i].uni_i = -1;
		time->robot[i].uni_j = -1;
		time->robot[i].red_k = -1;
		time->robot[i].min_ij = -1.0;
		time->robot[i].angle = -1.0;
		time->robot[i].centerx = -1.0;
		time->robot[i].centery = -1.0;
	}
}

void copiaEstadoTime(team* anterior_team, team* atual_team) {
	anterior_team->size = atual_team->size;
	for (int i = 0; i < atual_team->size; i++) {
		anterior_team->robot[i].id = atual_team->robot[i].id;
		anterior_team->robot[i].uni_i = atual_team->robot[i].uni_i;
		anterior_team->robot[i].uni_j = atual_team->robot[i].uni_j;
		anterior_team->robot[i].red_k = atual_team->robot[i].red_k;
		anterior_team->robot[i].min_ij = atual_team->robot[i].min_ij;
		anterior_team->robot[i].angle = atual_team->robot[i].angle;
		anterior_team->robot[i].centerx = atual_team->robot[i].centerx;
		anterior_team->robot[i].centery = atual_team->robot[i].centery;
	}
}

team detectaRobo(Mat& imagemProcessada, cluster_vec& clusterCorTime, cluster_vec& clusterComum) {
	team time;
	zeraTime(&time);
	float distance;
	relation r;

	int maiorDistanciaEntreCusterComumERed = 30;

	// cout << "clusterComum: " <<  clusterComum.len << endl;
	// cout << "clusterCOR: " <<  clusterCorTime.len << endl;
	for(int i = 0; i < clusterCorTime.len; i++) {
		// cout << "Tamanho cluster time: " <<  clusterCorTime.clus[i].size << endl;
		if (clusterCorTime.clus[i].size > MIN_CLUSTER_SIZE) {
			r.min = numeric_limits<float>::max();
			r.i = -1;
			r.j = -1;
			for (int j = 0; j < clusterComum.len; j++) {
				if (clusterComum.clus[j].size > MIN_CLUSTER_SIZE) {
					distance = EuclideanDistance(clusterCorTime.clus[i].centerx, clusterCorTime.clus[i].centery, clusterComum.clus[j].centerx, clusterComum.clus[j].centery);
					if (distance < r.min && distance < maiorDistanciaEntreCusterComumERed) {
						r.min = distance;
						//Posicao no clustar corTime
						r.i=i;
						//Posicao no cluster comum (red)
						r.j=j;
					}
				}
			}
			//Insere os ROBOS
			if (time.size < TEAM_SIZE) {
				int posicaoRobo = -1;
				for (int controleRobo = 0; controleRobo < time.size; controleRobo++) {
					if (time.robot[controleRobo].red_k == r.j) {
						posicaoRobo = controleRobo;
					}
				}
				//Nao existe um robo com o custer comum (red)
				if (posicaoRobo == -1) {
					//Insere um novo
					posicaoRobo = time.size;
					//Posicao no cluster cor do time
					time.robot[posicaoRobo].uni_i = r.i;
					time.robot[posicaoRobo].min_ij = -1;
				}
				//Registra a segunda relação entre entre o cluster cor do time e cluster comum (red)
				if (time.robot[posicaoRobo].red_k == r.j) {
					time.robot[posicaoRobo].uni_j = r.i;
					time.robot[posicaoRobo].min_ij = EuclideanDistance(clusterCorTime.clus[time.robot[posicaoRobo].uni_i].centerx, clusterCorTime.clus[time.robot[posicaoRobo].uni_i].centery, clusterCorTime.clus[time.robot[posicaoRobo].uni_j].centerx, clusterCorTime.clus[time.robot[posicaoRobo].uni_j].centery);;
				} else {
					//Posicao no cluster comum (red)
					time.robot[posicaoRobo].red_k = r.j;	
				}
				if (posicaoRobo == time.size) {
					//Insere um novo
					time.robot[posicaoRobo].id = time.size;
					time.size++;
				}
				if (time.robot[posicaoRobo].uni_j == -1) {
					time.robot[posicaoRobo].uni_j = time.robot[posicaoRobo].uni_i;
				}
			}
		}
	}

	//printf("Numero de robo: %i\n", time.size);
	for (int i = 0; i < time.size; i++) {
		// circle(imagemProcessada, cv::Point(clusterComum.clus[i].centerx, clusterComum.clus[i].centery), 20.0, cv::Scalar(0, 0, 255), 2, 8, 0);
		// line(imagemProcessada, cv::Point(clusterCorTime.clus[time.robot[i].uni_i].centerx, clusterCorTime.clus[time.robot[i].uni_i].centery), cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Scalar(0, 255, 0), 2, 8, 0);
		// line(imagemProcessada, cv::Point(clusterCorTime.clus[time.robot[i].uni_j].centerx, clusterCorTime.clus[time.robot[i].uni_j].centery), cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Scalar(0, 255, 0), 2, 8, 0);	

		// printf("ID: %i | cI: %i | cJ: %i | cRed: %i\n", time.robot[i].id, time.robot[i].uni_i, time.robot[i].uni_j, time.robot[i].red_k);


		time.robot[i].angle = 0;

		float leftx = clusterCorTime.clus[time.robot[i].uni_j].centerx;
		float rightx = clusterCorTime.clus[time.robot[i].uni_i].centerx;
		
		float bottomy=clusterCorTime.clus[time.robot[i].uni_j].centery;
		float uppery=clusterCorTime.clus[time.robot[i].uni_i].centery;
		
		if(clusterCorTime.clus[time.robot[i].uni_i].centerx < clusterCorTime.clus[time.robot[i].uni_j].centerx) {
			leftx=clusterCorTime.clus[time.robot[i].uni_i].centerx;
			rightx=clusterCorTime.clus[time.robot[i].uni_j].centerx;
		}
		if( clusterCorTime.clus[time.robot[i].uni_i].centery > clusterCorTime.clus[time.robot[i].uni_j].centery) {
			bottomy=clusterCorTime.clus[time.robot[i].uni_i].centery;
			uppery=clusterCorTime.clus[time.robot[i].uni_j].centery;
		}

		//Define o centro do Robo
		time.robot[i].centerx = leftx + ((rightx-leftx) / 2);
		time.robot[i].centery = uppery + ((bottomy-uppery) / 2);

		// printf("Robo: (%f, %f) | clusterComum: (%f, %f)\n", time.robot[i].centerx, time.robot[i].centery, clusterComum.clus[r.j].centerx, clusterComum.clus[r.j].centery);

		float diferencaX = clusterComum.clus[time.robot[i].red_k].centerx-time.robot[i].centerx;
		float diferencaY = clusterComum.clus[time.robot[i].red_k].centery-time.robot[i].centery;
		// printf("diferencaX: %f | diferencaY: %f\n", diferencaX, diferencaY);
		//X maior entao 90 ou 270
		
		line(imagemProcessada, cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Point(time.robot[i].centerx, time.robot[i].centery), cv::Scalar(0, 255, 0), 2, 8, 0);

		float hi = sqrt((diferencaX * diferencaX) + (diferencaY * diferencaY));
		float seno = 0;
		if (abs(diferencaY) > abs(diferencaX)) {
			if (hi > 0) {
				seno = asin(diferencaX / hi) * (180.0/3.14);
			} 
			if (diferencaY > 0) {
				time.robot[i].angle = 90 + seno;
			} else {
				time.robot[i].angle = 270 - seno;
			}
		} else {
		// 0 ou 180
			if (hi > 0) {
				seno = asin(diferencaY / hi) * (180.0/3.14);
			} 
			if (diferencaX < 0) {
				if (time.robot[i].angle +seno < 0) {
					time.robot[i].angle = 360 + seno;
				} else {
					time.robot[i].angle = seno;
				}
				
			} else {
				time.robot[i].angle = 180 - seno;
			}
		}
	}
	return time;
}

void writeNumber(string text, Mat img, int x, int y, cv::Scalar color)
{
	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 1;
	int thickness = 2;

	int baseline = 0;
	// Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;

	Point textOrg(x, y);
	// Point textOrg(x - textSize, y - textSize);
	putText(img, text, textOrg, fontFace, fontScale, color, thickness, 8);
}

void init_color(colors* c,float lowR,float highR,float lowG,float highG,float lowB,float highB)
{
  c->lowR=lowR;
  c->highR=highR;

  c->lowG=lowG;
  c->highG=highG;

  c->lowB=lowB;
  c->highB=highB;
}

void init_cluster(cluster* c,int xmin, int xmax, int ymin, int ymax,int size)
{
  c->size=size;
  c->xmin=xmin;
  c->xmax=xmax;
  c->ymin=ymin;
  c->ymax=ymax;
}

void check_neighbor(uint8_t* img,int i,int j,int n,int m, uint8_t* checked, cluster* c,colors* obj)
{
  if(c->size > MAX_CLUSTER_SIZE)
  {
	return;
  }

  if( checked[(i*m)+j]==0 )
  {
	//ONLY IF USING CHROMATICY
	// float avg=(img[(i*m*3+j*3)]+img[(i*m*3+j*3)+1]+img[(i*m*3+j*3)+2]);
	// float r=(float)img[(i*m*3+j*3)+2]/avg;
	// float g=(float)img[(i*m*3+j*3)+1]/avg;
	// float b=(float)img[(i*m*3+j*3)]/avg;

	//ONLY IF USING HSV
	float r=(float)img[(i*m*3+j*3)];
	float g=(float)img[(i*m*3+j*3)+1];
	float b=(float)img[(i*m*3+j*3)+2];
	if(
	  (r>=obj->lowR)&&
	  (r<=obj->highR)&&
	  (g>=obj->lowG)&&
	  (g<=obj->highG)&&
	  (b>=obj->lowB)&&
	  (b<=obj->highB) )
	{
	  checked[(i*m)+j]=1;
	  c->size=c->size+1;
	  if(j<c->xmin)
		c->xmin=j;
	  if(j>c->xmax)
		c->xmax=j;
	  if(i<c->ymin)
		c->ymin=i;
	  if(i>c->ymax)
		c->ymax=i;

	  if(i < n - 1)
		check_neighbor(img, i+1, j,n,m,checked,c,obj );
	  if(j < m - 1 )
		check_neighbor(img, i, j+1,n,m,checked,c,obj );
	  if(i > 0 )
		check_neighbor(img, i-1, j,n,m,checked,c,obj );
	  if(j > 0 )
		check_neighbor(img, i, j-1,n,m,checked,c,obj );
	}
  }
}

void threshold(Mat& image,colors* obj,Mat& frame, cluster_vec& c, int campoMin_X, int campoMin_Y, int campoMax_X, int campoMax_Y)
{
  uint8_t *img = image.data;
  int n=image.rows;
  int m=image.cols;

  uint8_t* checked=(uint8_t*)calloc(n*m,sizeof(uint8_t) );

  int counter=0;
  int maxCluster=c.len;
  for (int i = 0; i < maxCluster; i++)
  {
	init_cluster(&c.clus[i], campoMax_X, campoMin_X, campoMax_Y, campoMin_Y, 0);
  }


  for (int i = campoMin_Y; i < campoMax_Y; i++)
  {
	for (int j = campoMin_X; j < campoMax_X; j++)
	{
	  //ONLY IF USING CHROMATICY
	  // float avg=(img[(i*m*3+j*3)]+img[(i*m*3+j*3)+1]+img[(i*m*3+j*3)+2]);
	  // float r=(float)img[(i*m*3+j*3)+2]/avg;
	  // float g=(float)img[(i*m*3+j*3)+1]/avg;
	  // float b=(float)img[(i*m*3+j*3)]/avg;

	  //ONLY IF USING HSV
	  float r=(float)img[(i*m*3+j*3)];
	  float g=(float)img[(i*m*3+j*3)+1];
	  float b=(float)img[(i*m*3+j*3)+2];
	  if(
		(r>=obj->lowR)&&
		(r<=obj->highR)&&
		(g>=obj->lowG)&&
		(g<=obj->highG)&&
		(b>=obj->lowB)&&
		(b<=obj->highB) )
	  {
		if( checked[(i*m)+j]==0 )
		{

		  check_neighbor(img,i,j,n,m,checked,&c.clus[counter],obj);
		  if((c.clus[counter].size> MIN_CLUSTER_SIZE )&&(c.clus[counter].size< MAX_CLUSTER_SIZE))
		  {
			counter++;
			if(counter==maxCluster)
			{
			  maxCluster=maxCluster*2;
			  c.clus=(cluster*)realloc(c.clus,sizeof(cluster)*maxCluster);
			  for (int k = counter; k < maxCluster; k++)
			  {
				init_cluster(&c.clus[k],m,0,n,0,0);
			  }
			}
		  }
		  else
		  {
			init_cluster(&c.clus[counter],m,0,n,0,0);
		  }

		}
	  }
	}
  }

  for (int k = 0; k < counter; k++)
  {
	  c.clus[k].diffx=(c.clus[k].xmax-c.clus[k].xmin);
	  c.clus[k].diffy=(c.clus[k].ymax-c.clus[k].ymin);
	  c.clus[k].centerx=c.clus[k].xmin+(c.clus[k].diffx/2);
	  c.clus[k].centery=c.clus[k].ymin+(c.clus[k].diffy/2);
	  // cout <<"cluster["<<k<<"] size="<< c.clus[k].size;
	  // cout <<" centerx="<< c.clus[k].centerx;
	  // cout <<" centery="<< c.clus[k].centery;
	  // cout <<" diffx="<< c.clus[k].diffx;
	  // cout <<" diffy="<< c.clus[k].diffy<< endl;

	  rectangle(
		frame,
		cv::Point(c.clus[k].xmin, c.clus[k].ymin),
		cv::Point(c.clus[k].xmax, c.clus[k].ymax),
		//ONLY IF USING CHROMATICY
		// cv::Scalar(obj->highB*255, obj->highG*255, obj->highR*255)
		cv::Scalar(obj->highB, obj->highG, obj->highR)
	  );
  }

  c.len=maxCluster;
  free(checked);
}

//Faz um tracking meia boca
void tracking(Mat& imagemProcessada, team* anterior_team, team* atual_team) {
	// cout << "Time anterior: " << anterior_team->size << endl;
	// cout << "Time atual: " << atual_team->size << endl;

// printf("Atual (%.2f, %.2f) I: %i - ID: %i | Anterior (%.2f, %.2f) J: %i - ID: %i | Distancia: %f\n", 
// 				atual_team->robot[i].centerx, 
// 				atual_team->robot[i].centery, 
// 				i, 
// 				atual_team->robot[i].id, 
// 				anterior_team->robot[j].centerx, 
// 				anterior_team->robot[j].centery, 
// 				j, 
// 				anterior_team->robot[j].id, 
// 				distanceEntreRobot);
	// printf("--- Dados do time ANTERIOR ---\n");
 // 	for (int i = 0; i < anterior_team->size; i++) {
 // 		printf("INDICE: %i | ID: %i\n", i, anterior_team->robot[i].id);
 // 	}

 // 	printf("--- Dados do time ATUAL ---\n");
 // 	for (int i = 0; i < atual_team->size; i++) {
 // 		printf("INDICE: %i | ID: %i\n", i, atual_team->robot[i].id);
 // 	}

 	char controAtribuicao[atual_team->size];
 	for (int i = 0; i < atual_team->size; i++) {
 		controAtribuicao[i] = 0;
 	}

 	for (int i = 0; i < atual_team->size; i++) {
		float distance = numeric_limits<float>::max();
		int idAnterior = -1;
		for (int j = 0; j < anterior_team->size; j++) {
			float distanceEntreRobot = EuclideanDistance(atual_team->robot[i].centerx, atual_team->robot[i].centery, anterior_team->robot[j].centerx, anterior_team->robot[j].centery);
			// printf("(Indice: %i) - Atual ID: %i | (Indice: %i) - Anterior ID: %i | Distancia: %f\n", i, atual_team->robot[i].id, j, anterior_team->robot[j].id, distanceEntreRobot);
			if (distanceEntreRobot < distance) {
				distance = distanceEntreRobot;
				idAnterior = anterior_team->robot[j].id;
				controAtribuicao[i] = 1;
			}
		}
		// printf("--> OQUE ERA ID: %i | FOI ESCOLHIDO ID: %i\n", idAnterior, atual_team->robot[i].id);
		if (idAnterior > -1 && controAtribuicao[i] == 0) {
			atual_team->robot[i].id = idAnterior;
		}
		//line(imagemProcessada, cv::Point(anterior_team->robot[posicaoAnterior].centerx, anterior_team->robot[posicaoAnterior].centery), cv::Point(atual_team->robot[posicaoAtual].centerx, atual_team->robot[posicaoAtual].centery), cv::Scalar(0, 255, 0), 2, 8, 0);
	}
}

int main( int argc, char** argv ) {
	//OBTEM DO IHC
	// int campoMin_X = 50	;
	// int campoMin_Y = 18;
	// int campoMax_X = 460;
	// int campoMax_Y = 370;

	//FILE *arq;
	//arq = fopen("arq_IHC", "rt");	


	//Verificar esses valores
	float escalaPixel = (float(campoMax_X - campoMin_X) / FIELD_SIZE);
	//Escala em x
	float escalaX = float(FIELD_SIZE_X) / float(campoMax_X - campoMin_X);
	cout << "Escala em x: " << escalaX << endl;
	float escalaY = float(FIELD_SIZE_Y) / float(campoMax_Y - campoMin_Y);
	cout << "Escala em y: " << escalaY << endl;


	Mat cameraMatrix = (Mat_<double>(3,3) << 663.12266191,0.,298.54197567,0.,664.55319541,271.03745091,0.,0.,1.);
	Mat distCoeffs = (Mat_<double>(5,1) << -0.43831627,0.29401438,-0.00404992,-0.00459204,-0.12562864);
	Mat map1, map2;
	Rect roi;
	Mat NewCameraMtx = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size(640, 480), 1, Size(640, 480), &roi);
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), NewCameraMtx, Size(640, 480), CV_16SC2, map1, map2);

	//VideoCapture cap(argv[1]); // open the default camera
	VideoCapture cap(1); // open the default camera


	// check if we succeeded
	if(!cap.isOpened()) {
		return -1;
	}

	Mat imageTemp;

	cap >> imageTemp;
	remap(imageTemp, imageTemp, map1, map2, INTER_LINEAR);
	imageTemp = imageTemp(roi);
	imwrite("imagem_IHC.bmp", imageTemp);

	colors yellow;
	// init_color(&yellow,0.34f,1.0f,0.33f,1.0f,0.0f,0.32f);//CHROMATICY
	init_color(&yellow, 10,40, 10,90, 180,255);//HSV

	colors blue;
	// init_color(&blue,0.15f,0.35f,0.15f,0.35f,0.45f,1.0f);//CHROMATICY
	init_color(&blue, 100,120, 90,130, 180,255);//HSV

	colors red;
	// init_color(&red,0.4f,1.0f,0.1f,0.35f,0.1f,0.35f);//CHROMATICY
	init_color(&red, 140,190, 80,140, 170,255 );//HSV

	colors orange;
	init_color(&orange, 0,20, 130,180, 180,255 );//HSV


	cluster_vec cyellow;
	cyellow.len=EXPECTED_CLUSTERS;
	cyellow.clus=(cluster*)malloc(cyellow.len*sizeof(cluster));

	cluster_vec cblue;
	cblue.len=EXPECTED_CLUSTERS;
	cblue.clus=(cluster*)malloc(cblue.len*sizeof(cluster));

	cluster_vec cred;
	cred.len=EXPECTED_CLUSTERS;
	cred.clus=(cluster*)malloc(cred.len*sizeof(cluster));

	cluster_vec corange;
	corange.len=EXPECTED_CLUSTERS;
	corange.clus=(cluster*)malloc(corange.len*sizeof(cluster));

	team y_team;
	zeraTime(&y_team);
	team y_team_posicaoAnterior;
	zeraTime(&y_team_posicaoAnterior);

	team b_team;
	zeraTime(&b_team);
	team b_team_posicaoAnterior;
	zeraTime(&b_team_posicaoAnterior);

	int contaImagem = 0;
	for(;;) {
		printf("------ (%i) ------\n", contaImagem);

		cap >> imageTemp;

		remap(imageTemp, imageTemp, map1, map2, INTER_LINEAR);
		Mat image;
		imageTemp = imageTemp(roi);
		imageTemp.copyTo(image);
		cvtColor(image, image, COLOR_BGR2HSV);

		Mat frame(image.rows, image.cols, CV_8UC3, Scalar(0)); //IHC

		//Desenha campo
		rectangle(imageTemp, cv::Point(campoMin_X, campoMin_Y), cv::Point(campoMax_X, campoMax_Y), cv::Scalar(0, 0, 255));
		rectangle(frame, cv::Point(campoMin_X, campoMin_Y), cv::Point(campoMax_X, campoMax_Y), cv::Scalar(0, 0, 255));


		// cout<<"\033[2J\033[1;1H" << endl;
		// cout << "yellow" << endl;
		threshold(image, &yellow, frame, cyellow, campoMin_X, campoMin_Y, campoMax_X, campoMax_Y);

		// cout<<"blue"<<endl;
		threshold(image,&blue,frame,cblue, campoMin_X, campoMin_Y, campoMax_X, campoMax_Y);

		// cout<<"red"<<endl;
		threshold(image,&red,frame,cred, campoMin_X, campoMin_Y, campoMax_X, campoMax_Y);

		// cout<<"orange"<<endl;
		threshold(image,&orange,frame,corange, campoMin_X, campoMin_Y, campoMax_X, campoMax_Y);


		// cout<<"yellow"<<endl;
		y_team = detectaRobo(frame, cyellow, cred);
		// tracking(image, &y_team_posicaoAnterior, &y_team);
		
    	// cout << "blue" << endl;
		b_team = detectaRobo(frame, cblue, cred);
		// tracking(image, &b_team_posicaoAnterior, &b_team);

		for(int i=0;i<y_team.size;i++) {
			writeNumber(to_string(y_team.robot[i].id), imageTemp,y_team.robot[i].centerx,y_team.robot[i].centery,cv::Scalar(0,255,255) );
			writeNumber(to_string(y_team.robot[i].angle), frame,y_team.robot[i].centerx,y_team.robot[i].centery,cv::Scalar(0,255,255) );
		}

		for(int i=0;i<b_team.size;i++) {
			writeNumber(to_string(b_team.robot[i].id),imageTemp,b_team.robot[i].centerx,b_team.robot[i].centery,cv::Scalar(255,0,0) );
			writeNumber(to_string(b_team.robot[i].angle),frame,b_team.robot[i].centerx,b_team.robot[i].centery,cv::Scalar(255,0,0) );
		}

		circle(imageTemp, cv::Point(corange.clus[0].centerx, corange.clus[0].centery), 10.0, cv::Scalar(0, 0, 255), 2, 8, 0);

		cv::hconcat(imageTemp, frame, frame);
		imwrite("/home/fernando/Pictures/img/imagem_fer_"+to_string(contaImagem)+".jpg", frame);

		// add(imageTemp, frame, frame);//IHC

		//Mensagem
		cout << "-- yellow --" << endl;
		cout << y_team.robot[0].id << endl;
		cout << (y_team.robot[0].centerx * escalaX) << endl;
		cout << (y_team.robot[0].centery * escalaY) << endl;
		cout << y_team.robot[1].id << endl;
		cout << (y_team.robot[1].centerx * escalaX) << endl;
		cout << (y_team.robot[1].centery * escalaY) << endl;
		cout << y_team.robot[2].id << endl;
		cout << (y_team.robot[2].centerx * escalaX) << endl;
		cout << (y_team.robot[2].centery * escalaY) << endl;
    	cout << "-- blue --" << endl;
		cout << b_team.robot[0].id << endl;
		cout << (b_team.robot[0].centerx * escalaX) << endl;
		cout << (b_team.robot[0].centery * escalaY) << endl;
		cout << b_team.robot[1].id << endl;
		cout << (b_team.robot[1].centerx * escalaX) << endl;
		cout << (b_team.robot[1].centery * escalaY) << endl;
		cout << b_team.robot[2].id << endl;
		cout << (b_team.robot[2].centerx * escalaX) << endl;
		cout << (b_team.robot[2].centery * escalaY) << endl;
    	cout << "-- ball --" << endl;
		if (corange.clus[0].size == 0) {
			cout << "-1.0" << endl;
			cout << "-1.0" << endl;
		} else {
			cout << (corange.clus[0].centerx * escalaX) << endl;
			cout << (corange.clus[0].centery * escalaY) << endl;	
		}
		

		//Guarda a posicao anterior
		copiaEstadoTime(&y_team_posicaoAnterior, &y_team);
		copiaEstadoTime(&b_team_posicaoAnterior, &b_team);
		zeraTime(&y_team);
		zeraTime(&b_team);

		contaImagem++;
		imshow("edges", frame);//IHC
		if(waitKey(30) >= 0) {
			break;
		}
	}
	
	free(cyellow.clus);
	free(cblue.clus);
	free(cred.clus);
	cap.release();
	return 0;
}
