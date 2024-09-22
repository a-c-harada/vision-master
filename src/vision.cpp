#include<iostream>
#include<vector>
#include<ros/ros.h>
#include "vision/VisionToSituation.h"
#include "vision/VisionToAnalysis.h"
#include "vision/VisionToStrategy.h"
#include "vision/VisionToControl.h"
#include "ihc/IhcToVision.h"
#include<opencv2/highgui/highgui.hpp>

#include <stdlib.h>
#include <unistd.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits>
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include<thread>
#include<mutex>
#include<time.h>

using namespace cv;
using namespace std;

#define EXPECTED_CLUSTERS 16
#define MIN_CLUSTER_SIZE 25
#define MAX_CLUSTER_SIZE 400
#define TEAM_SIZE 3
#define FIELD_SIZE_X 1440
#define FIELD_SIZE_Y 1240
#define FRAME_BUFFER 6
// #define ROBOTRED1 150
// #define ROBOTRED2 100
// #define ROBOTRED3 50
#define LONGESTDISTANCE 20

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
	int counter;
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


void detectaRobo3(team* time,  cluster_vec& clusterUniform) {
	


	int smaller=3;
	if(clusterUniform.counter<smaller){
		smaller=clusterUniform.counter;
	}

	int aux[clusterUniform.counter];

	for(int i = 0; i < clusterUniform.counter; ++i){
		aux[i] = clusterUniform.clus[i].size;
	}

	//Ordena do maior para o menor
	int maiores[3] = {0,0,0};

	for(int i = 0; i < 3; ++i){
		int maior = 0;
		for (int j=1; j<clusterUniform.counter;j++){
			if(aux[j]> aux[maior]){
				maior = j;
			}
		}
		maiores[i] = maior;
		aux[maior] = 0;
	}

	for(int i=0;i<smaller;i++){
		time->size++;
		time->robot[i].id = i+1;
		time->robot[i].angle = -1;
		time->robot[i].centerx = clusterUniform.clus[maiores[i]].centerx;
		time->robot[i].centery = clusterUniform.clus[maiores[i]].centery;
	}

}

void detectaRobo2(team* time, Mat& imagemProcessada, cluster_vec& clusterCorTime, cluster_vec& clusterComum, int id) {
	// zeraTime(&time);
	float distance;
	relation r;

	int maiorDistanciaEntreCusterComumERed = LONGESTDISTANCE;

	cout << "Uniforme do robo: " <<  clusterCorTime.len << endl;
	cout << "Cor do time: " <<  clusterComum.len << endl;

	r.min = numeric_limits<float>::max();
	r.i = -1;
	r.j = -1;
	// for(int i = 0; i < clusterComum.len; i++) {
	// 	// cout << "Tamanho cluster time: " <<  clusterCorTime.clus[i].size << endl;
	// 	if (clusterComum.clus[i].size > MIN_CLUSTER_SIZE) {
	// 		for (int j = 0; j < clusterCorTime.len; j++) {
	// 			if (clusterCorTime.clus[j].size > MIN_CLUSTER_SIZE) {
			
	// 				distance = EuclideanDistance(clusterCorTime.clus[j].centerx, clusterCorTime.clus[j].centery, clusterComum.clus[i].centerx, clusterComum.clus[i].centery);
	// 				if (distance < r.min && distance < maiorDistanciaEntreCusterComumERed) {
	// 					r.min = distance;
	// 					//Posicao no cluster corTime (Uniforme)
	// 					r.i=i;
	// 					//Posicao no cluster comum (Cor do time)
	// 					r.j=j;
	// 				}
	// 			}
	// 		}
	// 	}
	// }
	for(int i = 0; i < clusterCorTime.counter; i++)
	{
		// if (clusterCorTime.clus[i].size > MIN_CLUSTER_SIZE)
		// {
			for(int j = 0; j < clusterComum.counter; j++)
			{
				// if (clusterComum.clus[j].size > MIN_CLUSTER_SIZE)
				// {
					distance = EuclideanDistance(clusterCorTime.clus[i].centerx, clusterCorTime.clus[i].centery, clusterComum.clus[j].centerx, clusterComum.clus[j].centery);
					if (distance < r.min && distance < maiorDistanciaEntreCusterComumERed) 
					{
						r.min = distance;
						//Posicao no cluster corTime (Uniforme)
						r.i=i;
						//Posicao no cluster comum (Cor do time)
						r.j=j;
					}
				// }
			}

		// }
	}


	//Insere os ROBOS
	time->robot[(id-1)].uni_i = r.i;
	time->robot[(id-1)].red_k = r.j;
	time->robot[(id-1)].min_ij = r.min;
	time->robot[(id-1)].id = id;
	time->size++;

	// circle(imagemProcessada, cv::Point(clusterComum.clus[i].centerx, clusterComum.clus[i].centery), 20.0, cv::Scalar(0, 0, 255), 2, 8, 0);
	// line(imagemProcessada, cv::Point(clusterCorTime.clus[time.robot[i].uni_i].centerx, clusterCorTime.clus[time.robot[i].uni_i].centery), cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Scalar(0, 255, 0), 2, 8, 0);
	// line(imagemProcessada, cv::Point(clusterCorTime.clus[time.robot[i].uni_j].centerx, clusterCorTime.clus[time.robot[i].uni_j].centery), cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Scalar(0, 255, 0), 2, 8, 0);	

	// printf("ID: %i | cI: %i | cJ: %i | cRed: %i\n", time.robot[i].id, time.robot[i].uni_i, time.robot[i].uni_j, time.robot[i].red_k);


	time->robot[(id-1)].angle = 0;
	// int bigger=0;
	// if(clusterCorTime.clus[time.robot[(id-1)].uni_j].size>clusterCorTime.clus[time.robot[i].uni_i].size){
	// 	bigger=time.robot[i].uni_j;
	// }
	// else{
	// 	bigger=time.robot[i].uni_i;
	// }


	// float leftx = clusterCorTime.clus[time.robot[i].uni_j].centerx;
	// float rightx = clusterCorTime.clus[time.robot[i].uni_i].centerx;
	
	// float bottomy=clusterCorTime.clus[time.robot[i].uni_j].centery;
	// float uppery=clusterCorTime.clus[time.robot[i].uni_i].centery;
	
	// if(clusterCorTime.clus[time.robot[i].uni_i].centerx < clusterCorTime.clus[time.robot[i].uni_j].centerx) {
	// 	leftx=clusterCorTime.clus[time.robot[i].uni_i].centerx;
	// 	rightx=clusterCorTime.clus[time.robot[i].uni_j].centerx;
	// }
	// if( clusterCorTime.clus[time.robot[i].uni_i].centery > clusterCorTime.clus[time.robot[i].uni_j].centery) {
	// 	bottomy=clusterCorTime.clus[time.robot[i].uni_i].centery;
	// 	uppery=clusterCorTime.clus[time.robot[i].uni_j].centery;
	// }

	// //Define o centro do Robo
	// time.robot[i].centerx = leftx + ((rightx-leftx) / 2);
	// time.robot[i].centery = uppery + ((bottomy-uppery) / 2);

	// //Define o centro do Robo
	time->robot[(id-1)].centerx = clusterCorTime.clus[time->robot[(id-1)].uni_i].centerx;
	time->robot[(id-1)].centery = clusterCorTime.clus[time->robot[(id-1)].uni_i].centery;


	// printf("Robo: (%f, %f) | clusterComum: (%f, %f)\n", time.robot[i].centerx, time.robot[i].centery, clusterComum.clus[r.j].centerx, clusterComum.clus[r.j].centery);

	float diferencaX = clusterComum.clus[time->robot[(id-1)].red_k].centerx-time->robot[(id-1)].centerx;
	float diferencaY = clusterComum.clus[time->robot[(id-1)].red_k].centery-time->robot[(id-1)].centery;
	// printf("diferencaX: %f | diferencaY: %f\n", diferencaX, diferencaY);
	//X maior entao 90 ou 270
	
	line(imagemProcessada, cv::Point(clusterComum.clus[time->robot[(id-1)].red_k].centerx, clusterComum.clus[time->robot[(id-1)].red_k].centery), cv::Point(time->robot[(id-1)].centerx, time->robot[(id-1)].centery), cv::Scalar(0, 255, 0), 2, 8, 0);

	float hi = sqrt((diferencaX * diferencaX) + (diferencaY * diferencaY));
	float seno = 0;
	if (abs(diferencaY) > abs(diferencaX)) {
		if (hi > 0) {
			seno = asin(diferencaX / hi) * (180.0/3.14);
			// seno = asin(diferencaX / hi);
		} 
		if (diferencaY > 0) {
			time->robot[(id-1)].angle = 90 + seno;
		} else {
			time->robot[(id-1)].angle = 270 - seno;
		}
	} else {
	// 0 ou 180
		if (hi > 0) {
			seno = asin(diferencaY / hi) * (180.0/3.14);
			// seno = asin(diferencaY / hi);
		} 
		if (diferencaX < 0) {
			if (time->robot[(id-1)].angle +seno < 0) {
				time->robot[(id-1)].angle = 360 + seno;
			} else {
				time->robot[(id-1)].angle = seno;
			}
			
		} else {
			time->robot[(id-1)].angle = 180 - seno;
		}
	}
	// time.robot[i].angle/= (180.0/3.14);//SEND IN RADIANS

	// return time;
}


void detectaRobo(team* time, Mat& imagemProcessada, cluster_vec& clusterCorTime, cluster_vec& clusterComum, int id) {
	// zeraTime(&time);
	float distance;
	relation r;

	int maiorDistanciaEntreCusterComumERed = LONGESTDISTANCE;

	cout << "Uniforme do robo: " <<  clusterCorTime.len << endl;
	cout << "Cor do time: " <<  clusterComum.len << endl;
	for(int i = 0; i < clusterComum.len; i++) {
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
			if (time->size < TEAM_SIZE) {
				int posicaoRobo = -1;
				for (int controleRobo = 0; controleRobo < time->size; controleRobo++) {
					if (time->robot[controleRobo].red_k == r.j) {
						posicaoRobo = controleRobo;
						// cout<< "posicaoRobo=" << posicaoRobo <<end;
						// cout<< "r=" << clusterComum.clus[j] <<end;
					}
				}
				//Nao existe um robo com o custer comum (red)
				if (posicaoRobo == -1) {
					//Insere um novo
					posicaoRobo = time->size;
					//Posicao no cluster cor do time
					time->robot[posicaoRobo].uni_i = r.i;
					time->robot[posicaoRobo].min_ij = -1;
				}
				//Registra a segunda relação entre entre o cluster cor do time e cluster comum (red)
				// if (time.robot[posicaoRobo].red_k == r.j) {
				// 	time.robot[posicaoRobo].uni_j = r.i;
				// 	time.robot[posicaoRobo].min_ij = EuclideanDistance(clusterCorTime.clus[time.robot[posicaoRobo].uni_i].centerx, clusterCorTime.clus[time.robot[posicaoRobo].uni_i].centery, clusterCorTime.clus[time.robot[posicaoRobo].uni_j].centerx, clusterCorTime.clus[time.robot[posicaoRobo].uni_j].centery);;
				// } else {
				// 	//Posicao no cluster comum (red)
				// 	time.robot[posicaoRobo].red_k = r.j;	
				// }
				//Registra a segunda relação entre entre o cluster cor do time e cluster comum (red)
				time->robot[posicaoRobo].red_k = r.j;
				if (posicaoRobo == time->size) {
					//Insere um novo

					// time->robot[posicaoRobo].id = time->size;
					time->robot[posicaoRobo].id = id;
					time->size++;
				}
				if (time->robot[posicaoRobo].uni_j == -1) {
					time->robot[posicaoRobo].uni_j = time->robot[posicaoRobo].uni_i;
				}
			}
		}
	}

	//printf("Numero de robo: %i\n", time->size);
	for (int i = 0; i < time->size; i++) {
		// circle(imagemProcessada, cv::Point(clusterComum.clus[i].centerx, clusterComum.clus[i].centery), 20.0, cv::Scalar(0, 0, 255), 2, 8, 0);
		// line(imagemProcessada, cv::Point(clusterCorTime.clus[time.robot[i].uni_i].centerx, clusterCorTime.clus[time.robot[i].uni_i].centery), cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Scalar(0, 255, 0), 2, 8, 0);
		// line(imagemProcessada, cv::Point(clusterCorTime.clus[time.robot[i].uni_j].centerx, clusterCorTime.clus[time.robot[i].uni_j].centery), cv::Point(clusterComum.clus[time.robot[i].red_k].centerx, clusterComum.clus[time.robot[i].red_k].centery), cv::Scalar(0, 255, 0), 2, 8, 0);	

		// printf("ID: %i | cI: %i | cJ: %i | cRed: %i\n", time.robot[i].id, time.robot[i].uni_i, time.robot[i].uni_j, time.robot[i].red_k);


		time->robot[i].angle = 0;
		// int bigger=0;
		// if(clusterCorTime.clus[time.robot[i].uni_j].size>clusterCorTime.clus[time.robot[i].uni_i].size){
		// 	bigger=time.robot[i].uni_j;
		// }
		// else{
		// 	bigger=time.robot[i].uni_i;
		// }


		// float leftx = clusterCorTime.clus[time.robot[i].uni_j].centerx;
		// float rightx = clusterCorTime.clus[time.robot[i].uni_i].centerx;
		
		// float bottomy=clusterCorTime.clus[time.robot[i].uni_j].centery;
		// float uppery=clusterCorTime.clus[time.robot[i].uni_i].centery;
		
		// if(clusterCorTime.clus[time.robot[i].uni_i].centerx < clusterCorTime.clus[time.robot[i].uni_j].centerx) {
		// 	leftx=clusterCorTime.clus[time.robot[i].uni_i].centerx;
		// 	rightx=clusterCorTime.clus[time.robot[i].uni_j].centerx;
		// }
		// if( clusterCorTime.clus[time.robot[i].uni_i].centery > clusterCorTime.clus[time.robot[i].uni_j].centery) {
		// 	bottomy=clusterCorTime.clus[time.robot[i].uni_i].centery;
		// 	uppery=clusterCorTime.clus[time.robot[i].uni_j].centery;
		// }

		// //Define o centro do Robo
		// time.robot[i].centerx = leftx + ((rightx-leftx) / 2);
		// time.robot[i].centery = uppery + ((bottomy-uppery) / 2);

		// //Define o centro do Robo
		time->robot[i].centerx = clusterCorTime.clus[time->robot[i].uni_i].centerx;
		time->robot[i].centery = clusterCorTime.clus[time->robot[i].uni_i].centery;


		// printf("Robo: (%f, %f) | clusterComum: (%f, %f)\n", time.robot[i].centerx, time.robot[i].centery, clusterComum.clus[r.j].centerx, clusterComum.clus[r.j].centery);

		float diferencaX = clusterComum.clus[time->robot[i].red_k].centerx-time->robot[i].centerx;
		float diferencaY = clusterComum.clus[time->robot[i].red_k].centery-time->robot[i].centery;
		// printf("diferencaX: %f | diferencaY: %f\n", diferencaX, diferencaY);
		//X maior entao 90 ou 270
		
		line(imagemProcessada, cv::Point(clusterComum.clus[time->robot[i].red_k].centerx, clusterComum.clus[time->robot[i].red_k].centery), cv::Point(time->robot[i].centerx, time->robot[i].centery), cv::Scalar(0, 255, 0), 2, 8, 0);

		float hi = sqrt((diferencaX * diferencaX) + (diferencaY * diferencaY));
		float seno = 0;
		if (abs(diferencaY) > abs(diferencaX)) {
			if (hi > 0) {
				seno = asin(diferencaX / hi) * (180.0/3.14);
				// seno = asin(diferencaX / hi);
			} 
			if (diferencaY > 0) {
				time->robot[i].angle = 90 + seno;
			} else {
				time->robot[i].angle = 270 - seno;
			}
		} else {
		// 0 ou 180
			if (hi > 0) {
				seno = asin(diferencaY / hi) * (180.0/3.14);
				// seno = asin(diferencaY / hi);
			} 
			if (diferencaX < 0) {
				if (time->robot[i].angle +seno < 0) {
					time->robot[i].angle = 360 + seno;
				} else {
					time->robot[i].angle = seno;
				}
				
			} else {
				time->robot[i].angle = 180 - seno;
			}
		}
		// time.robot[i].angle/= (180.0/3.14);//SEND IN RADIANS
	}
	// return time;
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
			// cout<< "hue=" << r << " sat=" << g << " value="<< b << endl;
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
	  // cout <<"cluster["<<k<<"].size="<< c.clus[k].size <<endl;
	  // cout <<" centerx="<< c.clus[k].centerx;
	  // cout <<" centery="<< c.clus[k].centery;
	  // cout <<" diffx="<< c.clus[k].diffx;
	  // cout <<" diffy="<< c.clus[k].diffy<< endl;

		Mat point(1,1, CV_8UC3, Scalar(obj->highR,obj->highG,obj->highB));
		cvtColor(point, point, COLOR_HSV2BGR);
		rectangle(
			frame,
			cv::Point(c.clus[k].xmin, c.clus[k].ymin),
			cv::Point(c.clus[k].xmax, c.clus[k].ymax),
		//ONLY IF USING CHROMATICY
		// cv::Scalar(obj->highB*255, obj->highG*255, obj->highR*255)
			cv::Scalar(point.data[0], point.data[1], point.data[2])
			);
	}
	// cout<<endl;
	c.len=maxCluster;
	c.counter=counter;
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

void lecor(char nameFile[],
			colors& yellow,
			colors& blue,
			colors& red,
			colors& green,
			colors& purple,
			colors& orange)
{
	//Cor do time
	// colors yellow;

	char str[1000];
	FILE * arq;
	arq = fopen(nameFile, "r");	
	//-----yellow---
	int low_H_y=0;
	int low_S_y=0;
	int low_V_y=0;
	int high_H_y=0;
	int high_S_y=0;
	int high_V_y=0;
    //-----blue-----
	int low_H_b=0;
	int low_S_b=0;
	int low_V_b=0;
	int high_H_b=0;
	int high_S_b=0;
	int high_V_b=0;
	//-----red-------
	int low_H_r=0;
	int low_S_r=0;
	int low_V_r=0;
	int high_H_r=0;
	int high_S_r=0;
	int high_V_r=0;
	//-----green-----
	int low_H_g=0;
	int low_S_g=0;
	int low_V_g=0;
	int high_H_g=0;
	int high_S_g=0;
	int high_V_g=0;
	//-----purple----
	int low_H_p=0;
	int low_S_p=0;
	int low_V_p=0;
	int high_H_p=0;
	int high_S_p=0;
	int high_V_p=0;
	//-----orange-----
	int low_H_o=0;
	int low_S_o=0;
	int low_V_o=0;
	int high_H_o=0;
	int high_S_o=0;
	int high_V_o=0;

	while (!feof(arq)) {
		fscanf(arq, "%s", str);

		// 
		if (strcmp(str, "yellow") == 0) {
			fscanf(arq, "%d %d %d %d %d %d", &low_H_y,&high_H_y, &low_S_y,&high_S_y, &low_V_y,&high_V_y  );
		}
		// 
		if (strcmp(str, "blue") == 0) {
			fscanf(arq, "%d %d %d %d %d %d", &low_H_b,&high_H_b, &low_S_b,&high_S_b, &low_V_b,&high_V_b  );
		}
		// 
		if (strcmp(str, "red") == 0) {
			fscanf(arq, "%d %d %d %d %d %d", &low_H_r,&high_H_r, &low_S_r,&high_S_r, &low_V_r,&high_V_r  );
		}
		//
		if (strcmp(str, "green") == 0) {
			fscanf(arq, "%d %d %d %d %d %d", &low_H_g,&high_H_g, &low_S_g,&high_S_g, &low_V_g,&high_V_g  );
		}
		//
		if (strcmp(str, "purple") == 0) {
			fscanf(arq, "%d %d %d %d %d %d", &low_H_p,&high_H_p, &low_S_p,&high_S_p, &low_V_p,&high_V_p  );
		}
		//
		if (strcmp(str, "orange") == 0) {
			fscanf(arq, "%d %d %d %d %d %d", &low_H_o,&high_H_o, &low_S_o,&high_S_o, &low_V_o,&high_V_o  );
		}

	}

	fclose(arq);

	//Cor do time
	init_color(&yellow, low_H_y,high_H_y, low_S_y,high_S_y, low_V_y,high_V_y );//HSV

	//Cor do time
	init_color(&blue, low_H_b,high_H_b, low_S_b,high_S_b, low_V_b,high_V_b );//HSV

	//Uniforme robo 1
	init_color(&red, low_H_r,high_H_r, low_S_r,high_S_r, low_V_r,high_V_r );//HSV

	//Uniforme robo 2
	init_color(&green,  low_H_g,high_H_g, low_S_g,high_S_g, low_V_g,high_V_g  );//HSV

	//Uniforme robo 3
	init_color(&purple, low_H_p,high_H_p, low_S_p,high_S_p, low_V_p,high_V_p );//HSV	-> pink!!!

	//Ball
	init_color(&orange, low_H_o,high_H_o, low_S_o,high_S_o, low_V_o,high_V_o );//HSV
}

void achaDelimitacaoDeCampo(char nameFile[], 
		int* campoMin_X, 
		int* campoMin_Y, 
		int* campoMax_X, 
		int* campoMax_Y, 
		int* areaDefSup_X, 
		int* areaDefSup_Y, 
		int* areaDefInf_X, 
		int* areaDefInf_Y, 
		int* areaAtaSup_X, 
		int* areaAtaSup_Y, 
		int* areaAtaInf_X, 
		int* areaAtaInf_Y,
		int* golAtaSup_X,
		int* golAtaSup_Y,
		int* golAtaInf_X,
		int* golAtaInf_Y,
		int* golDefSup_X,
		int* golDefSup_Y,
		int* golDefInf_X,
		int* golDefInf_Y
	){
	char str[1000];
	FILE * arq;
	arq = fopen(nameFile, "r");	
	
	while (!feof(arq)) {
		fscanf(arq, "%s", str);

		// Campo Superior Esquerdo
		if (strcmp(str, "cpose") == 0) {
			fscanf(arq, "%d", campoMin_X);
			fscanf(arq, "%d", campoMin_Y);
		}

		// Campo Superior Direito
		if (strcmp(str, "cposd") == 0) {
			int campoSupDir_X, campoSupDir_Y;
			fscanf(arq, "%d", &campoSupDir_X);
			fscanf(arq, "%d", &campoSupDir_Y);
			cout << "campo superior direito " << campoSupDir_X << " " << campoSupDir_Y << endl; 
		}

		// // Campo Inferior Esquerdo
		// if (strcmp(str, "cpoie") == 0) {

		// }

		// Campo Inferior Direito
		if (strcmp(str, "cpoid") == 0) {
			fscanf(arq, "%d", campoMax_X);
			fscanf(arq, "%d", campoMax_Y);
		}

		// // Area Defesa Superior Esquerdo
		// if (strcmp(str, "adse") == 0) {

		// }

		// Area Defesa Superior Direto
		if (strcmp(str, "adsd") == 0) {
			fscanf(arq, "%d", areaDefSup_X);
			fscanf(arq, "%d", areaDefSup_Y);
		}

		// Area Defesa Inferior Esquerdo
		if (strcmp(str, "adie") == 0) {
			fscanf(arq, "%d", areaDefInf_X);
			fscanf(arq, "%d", areaDefInf_Y);
		}

		// // Area Defesa Inferior Direito
		// if (strcmp(str, "adid") == 0) {

		// }

		// // Area Ataque Superior Direito
		// if (strcmp(str, "aasd") == 0) {

		// }

		// Area Ataque Superior Esquerdo
		if (strcmp(str, "aase") == 0) {
			fscanf(arq, "%d", areaAtaSup_X);
			fscanf(arq, "%d", areaAtaSup_Y);
		}

		// Area Ataque Inferior Direito
		if (strcmp(str, "aaid") == 0) {
			fscanf(arq, "%d", areaAtaInf_X);
			fscanf(arq, "%d", areaAtaInf_Y);
		}

		// // Area Ataque Inferior Esquerdo
		// if (strcmp(str, "aaie") == 0) {

		// }

		// Gol Defesa Superior
		if (strcmp(str, "gds") == 0) {
			fscanf(arq, "%d", golDefSup_X);
			fscanf(arq, "%d", golDefSup_Y);
		}

		// Gol Defesa Inferior		
		if (strcmp(str, "gdi") == 0) {
			fscanf(arq, "%d", golDefInf_X);
			fscanf(arq, "%d", golDefInf_Y);
		}

		// Gol Ataque Superior
		if (strcmp(str, "gas") == 0) {
			fscanf(arq, "%d", golAtaSup_X);
			fscanf(arq, "%d", golAtaSup_Y);

		}

		// Gol Ataque Inferior
		if (strcmp(str, "gai") == 0) {
			fscanf(arq, "%d", golAtaInf_X);
			fscanf(arq, "%d", golAtaInf_Y);
		}
	}
	fclose(arq);
}

//Faz um tracking meia boca
void naoEumTrackingDeVerdade(
team* anterior_team, team* atual_team) {
	for (int i = 0; i < atual_team->size; i++) {
		if (atual_team->robot[i].centerx == -1.0) {
			atual_team->robot[i].id = anterior_team->robot[i].id;
			atual_team->robot[i].uni_i = anterior_team->robot[i].uni_i;
			atual_team->robot[i].uni_j = anterior_team->robot[i].uni_j;
			atual_team->robot[i].red_k = anterior_team->robot[i].red_k;
			atual_team->robot[i].min_ij = anterior_team->robot[i].min_ij;
			atual_team->robot[i].angle = anterior_team->robot[i].angle;
			atual_team->robot[i].centerx = anterior_team->robot[i].centerx;
			atual_team->robot[i].centery = anterior_team->robot[i].centery;
		}
	}
}



void orderRobots(team *t, cluster_vec reds)
{
	// team temp;
	Robot temp;

	cout << "cluster vermelho 0: " << reds.clus[t->robot[0].red_k].size << endl;
	cout << "cluster vermelho 1: " << reds.clus[t->robot[1].red_k].size << endl;
	cout << "cluster vermelho 2: " << reds.clus[t->robot[2].red_k].size << endl;

	// // Identifica id 1
	// int min1, max1;
	// min1= 20; max1= 55;
	// if ((reds.clus[t->robot[0].red_k].size >= min1) and (reds.clus[t->robot[0].red_k].size <= max1)){
	// 	t->robot[0].id=1;
	// }
	// else if ((reds.clus[t->robot[1].red_k].size >= min1) and (reds.clus[t->robot[1].red_k].size <= max1)){
	// 	t->robot[1].id=1;
	// }
	// else if ((reds.clus[t->robot[2].red_k].size >= min1) and (reds.clus[t->robot[2].red_k].size <= max1)){
	// 	t->robot[2].id=1;
	// }

	// // Identifica id 2
	// int min2, max2;
	// min2= 70; max2= 140;
	// if ((reds.clus[t->robot[0].red_k].size >= min2) and (reds.clus[t->robot[0].red_k].size <= max2)){
	// 	t->robot[0].id=2;
	// }
	// else if ((reds.clus[t->robot[1].red_k].size >= min2) and (reds.clus[t->robot[1].red_k].size <= max2)){
	// 	t->robot[1].id=2;
	// }
	// else if ((reds.clus[t->robot[2].red_k].size >= min2) and (reds.clus[t->robot[2].red_k].size <= max2)){
	// 	t->robot[2].id=2;
	// }

	// // Identifica id 3
	// int min3, max3;
	// min3= 160; max3= 190;
	// if ((reds.clus[t->robot[0].red_k].size >= min3) and (reds.clus[t->robot[0].red_k].size <= max3)){
	// 	t->robot[0].id=3;
	// }
	// else if ((reds.clus[t->robot[1].red_k].size >= min3) and (reds.clus[t->robot[1].red_k].size <= max3)){
	// 	t->robot[1].id=3;
	// }
	// else if ((reds.clus[t->robot[2].red_k].size >= min3) and (reds.clus[t->robot[2].red_k].size <= max3)){
	// 	t->robot[2].id=3;
	// }	


	if( reds.clus[t->robot[0].red_k].size > reds.clus[t->robot[1].red_k].size )
	{
		temp=t->robot[1];
		t->robot[1]=t->robot[0];
		t->robot[0]=temp;
	}
	if( reds.clus[t->robot[1].red_k].size > reds.clus[t->robot[2].red_k].size )
	{
		temp=t->robot[2];
		t->robot[2]=t->robot[1];
		t->robot[1]=temp;
	}
	if( reds.clus[t->robot[0].red_k].size > reds.clus[t->robot[1].red_k].size )
	{
		temp=t->robot[1];
		t->robot[1]=t->robot[0];
		t->robot[0]=temp;
	}

	t->robot[0].id=1;
	t->robot[1].id=2;
	t->robot[2].id=3;

	// cout << "robot[0] with red.size="<<reds.clus[t->robot[0].red_k].size<<endl;
	// cout << "robot[1] with red.size="<<reds.clus[t->robot[1].red_k].size<<endl;
	// cout << "robot[2] with red.size="<<reds.clus[t->robot[2].red_k].size<<endl;

}

int findBall(cluster_vec corange)
{
	cout<<"corange.counter="<<corange.counter<<endl;
	int max=0;
	int index=0;
	for(int i=0;i<corange.counter;i++)
	{
		if(corange.clus[i].size>max){
			max=corange.clus[i].size;
			index=i;
		}
	}
	return index;
}

int our_team = 0;

void readIhcVision(const ihc::IhcToVision& msg){
    //ROS_INFO("I heard: [%s]", msg->data.c_str());
    std::cout << "=============================================================\n";
    std::cout << msg.ourColor << std::endl; 
    std::cout << msg.color << std::endl;
    our_team = msg.color;
    std::cout << our_team << std::endl; 
    std::cout << "=============================================================\n";
    

}

int main(int argc, char **argv){
	//Mat originalImg;
	//Mat imageTemp;
	Mat frames[FRAME_BUFFER];

	//thread th1(readImage,&originalImg);
	int a,b,c;

	// sleep(1);

	ros::init(argc,argv,"vision");


	ros::NodeHandle n;

	ros::Publisher pubToSituation = 
	n.advertise<vision::VisionToSituation>("vision_to_situation",1);
	ros::Publisher pubToAnalysis = 
	n.advertise<vision::VisionToAnalysis>("vision_to_analysis",10000);
	ros::Publisher pubToStrategy = 
	n.advertise<vision::VisionToStrategy>("vision_to_strategy",10000);
	ros::Publisher pubToControl = 
	n.advertise<vision::VisionToControl>("vision_to_control",10000);
	
	ros::Subscriber sub = n.subscribe("ihc_to_vision", 1000, readIhcVision);

	vision::VisionToSituation msgToSituation;
	vision::VisionToAnalysis msgToAnalysis;
	vision::VisionToStrategy msgToStrategy;
	vision::VisionToControl msgToControl;
	/*-------------------------------------------------------------------------*/
	//OBTEM DO IHC
	int campoMin_X = 0;
	int campoMin_Y = 0;
	int campoMax_X = 0;
	int campoMax_Y = 0;
	int areaDefSup_X = 0;
	int areaDefSup_Y = 0;
	int areaDefInf_X = 0;
	int areaDefInf_Y = 0;
	int areaAtaSup_X = 0;
	int areaAtaSup_Y = 0;
	int areaAtaInf_X = 0;
	int areaAtaInf_Y = 0;
	int golAtaSup_X = 0;
	int golAtaSup_Y = 0;
	int golAtaInf_X = 0;
	int golAtaInf_Y = 0;
	int golDefSup_X = 0;
	int golDefSup_Y = 0;
	int golDefInf_X = 0;
	int golDefInf_Y = 0;

	char nameFile[] = "/home/capivara/catkin_ws/src/IHC-JAVA/arq_IHC.txt";
	char colorFile[] = "/home/capivara/catkin_ws/src/IHC-JAVA/color.txt";

	colors yellow;
	colors blue;
	colors red;
	colors green;
	colors purple;
	colors orange;

	lecor(colorFile,yellow,blue,red,green,purple,orange);
	// char nameFile[] = "arq_IHC.txt";
	achaDelimitacaoDeCampo(nameFile, 
		&campoMin_X, 
		&campoMin_Y, 
		&campoMax_X, 
		&campoMax_Y, 
		&areaDefSup_X, 
		&areaDefSup_Y, 
		&areaDefInf_X, 
		&areaDefInf_Y, 
		&areaAtaSup_X, 
		&areaAtaSup_Y, 
		&areaAtaInf_X, 
		&areaAtaInf_Y,
		&golAtaSup_X,
		&golAtaSup_Y,
		&golAtaInf_X,
		&golAtaInf_Y,
		&golDefSup_X,
		&golDefSup_Y,
		&golDefInf_X,
		&golDefInf_Y
		);

	cout << "Delimitacoes de Campo" << endl;
	// printf("campoMax_X: %i\n", campoMax_X);
	// printf("campoMin_X: %i\n", campoMin_X);
	// printf("campoMax_Y: %i\n", campoMax_Y);
	// printf("campoMin_Y: %i\n", campoMin_Y);
	cout << "Campo superior esquerdo: " << campoMin_X << " " << campoMin_Y << endl;
	cout << "Campo inferior direito: " << campoMax_X << " " << campoMax_Y << endl;
	cout << "Area defesa superior: " << areaDefSup_X << " " << areaDefSup_Y << endl;
	cout << "Area defesa inferior: " << areaDefInf_X << " " << areaDefInf_Y << endl;
	cout << "Area ataque superior: " << areaAtaSup_X << " " << areaAtaSup_Y << endl;
	cout << "Area ataque inferior: " << areaAtaInf_X << " " << areaAtaInf_Y << endl;

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

	//VideoCapture cap("pre.mp4");  //argv[1]); // open the default camera

	VideoCapture cap(0);
	//VideoCapture cap(1); // open the default camera

	// check if we succeeded
	if(!cap.isOpened()) {
		return -1;
	}

	Mat imageTemp;

	for (int i = 0; i < FRAME_BUFFER; ++i)
	{
		cap >> frames[i];
	}
	imageTemp=frames[0];

	//img_mutex.lock();
	remap(imageTemp, imageTemp, map1, map2, INTER_LINEAR);
	//img_mutex.unlock();
	imageTemp = imageTemp(roi);
	imwrite("/home/capivara/catkin_ws/src/vision/src/assets/imagem_IHC.png", imageTemp);
	// imwrite("imagem_IHC.bmp", imageTemp);


	

	cluster_vec cyellow;
	cyellow.len=EXPECTED_CLUSTERS;
	cyellow.clus=(cluster*)malloc(cyellow.len*sizeof(cluster));

	cluster_vec cblue;
	cblue.len=EXPECTED_CLUSTERS;
	cblue.clus=(cluster*)malloc(cblue.len*sizeof(cluster));

	cluster_vec cred;
	cred.len=EXPECTED_CLUSTERS*2;
	cred.clus=(cluster*)malloc(cred.len*sizeof(cluster));

	cluster_vec cgreen;
	cgreen.len=EXPECTED_CLUSTERS*2;
	cgreen.clus=(cluster*)malloc(cgreen.len*sizeof(cluster));

	cluster_vec cpurple;
	cpurple.len=EXPECTED_CLUSTERS*2;
	cpurple.clus=(cluster*)malloc(cpurple.len*sizeof(cluster));

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
	time_t timestamp;
	/*--------------------------------------------------------------------------*/

	ros::Rate loop_rate(100);
	while(ros::ok())
	{
  	// int fps = cap.get(CAP_PROP_FPS);
  	// int multiplier = fps * seconds;

	   	printf("------ (%i) ------\n", contaImagem);
		for (int i = 0; i < 6; ++i)
		{
			cap >> frames[i];
		}
		imageTemp=frames[0];
		// flip(imageTemp,imageTemp,-1);

		//img_mutex.lock();
		remap(imageTemp, imageTemp, map1, map2, INTER_LINEAR);
		//img_mutex.unlock();
		Mat image;
		imageTemp = imageTemp(roi);
		imageTemp.copyTo(image);
		cvtColor(image, image, COLOR_BGR2HSV);
		Mat frame(image.rows, image.cols, CV_8UC3, Scalar(0)); //IHC
		//Desenha campo
		rectangle(imageTemp, cv::Point(campoMin_X, campoMin_Y), cv::Point(campoMax_X, campoMax_Y), cv::Scalar(0, 0, 255));
		rectangle(imageTemp, cv::Point(areaDefSup_X, areaDefSup_Y), cv::Point(areaDefInf_X, areaDefInf_Y), cv::Scalar(0, 255, 0));
		rectangle(imageTemp, cv::Point(areaAtaSup_X, areaAtaSup_Y), cv::Point(areaAtaInf_X, areaAtaInf_Y), cv::Scalar(0, 255, 0));
		line(imageTemp, cv::Point(golAtaSup_X, golAtaSup_Y), cv::Point(golAtaInf_X, golAtaInf_Y),  cv::Scalar(255, 0, 0));
		line(imageTemp, cv::Point(golDefSup_X, golDefSup_Y), cv::Point(golDefInf_X, golDefInf_Y),  cv::Scalar(255, 0, 0));
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

		// cout<<"green"<<endl;
		threshold(image,&green,frame,cgreen, campoMin_X, campoMin_Y, campoMax_X, campoMax_Y);

		// cout<<"purple"<<endl;
		threshold(image,&purple,frame,cpurple, campoMin_X, campoMin_Y, campoMax_X, campoMax_Y);


		//int our_team= 2; // ***** RECEBER DA IHC *****
		
		if (our_team == 1){ // our team is yellow		
			// Our robots
			// cout<<"yellow"<<endl;
			zeraTime(&y_team);
			detectaRobo2(&y_team, frame, cred, cyellow, 1);
			detectaRobo2(&y_team, frame, cgreen, cyellow, 2);
			detectaRobo2(&y_team, frame, cpurple, cyellow, 3);

			//Rival robots
			zeraTime(&b_team);
			detectaRobo3(&b_team, cblue);
		}
		
		else if (our_team == 0){ // our team is blue
			// Our robots
			// cout << "blue" << endl;
			zeraTime(&b_team);
			detectaRobo2(&b_team, frame, cred, cblue, 1);
			detectaRobo2(&b_team, frame, cgreen, cblue, 2);
			detectaRobo2(&b_team, frame, cpurple, cblue, 3);

			//Rival robots
			zeraTime(&y_team);
			detectaRobo3(&y_team, cyellow);
		}

		// tracking(image, &y_team_posicaoAnterior, &y_team);
		naoEumTrackingDeVerdade(&y_team_posicaoAnterior, &y_team);
		naoEumTrackingDeVerdade(&b_team_posicaoAnterior, &b_team);

		// orderRobots(&y_team,cred);
		// orderRobots(&b_team,cred);
		int ball=findBall(corange);

		// tracking(image, &b_team_posicaoAnterior, &b_team);
		for(int i=0;i<y_team.size;i++) {
			writeNumber(to_string(y_team.robot[i].id), imageTemp, y_team.robot[i].centerx, y_team.robot[i].centery, cv::Scalar(0,255,255) );
			writeNumber(to_string( int(y_team.robot[i].angle)), frame, y_team.robot[i].centerx, y_team.robot[i].centery, cv::Scalar(0,255,255) );
		}
		for(int i=0;i<b_team.size;i++) {
			writeNumber(to_string(b_team.robot[i].id),imageTemp,b_team.robot[i].centerx,b_team.robot[i].centery,cv::Scalar(255,0,0) );
			writeNumber(to_string( int(b_team.robot[i].angle)),frame,b_team.robot[i].centerx,b_team.robot[i].centery,cv::Scalar(255,0,0) );
		}
		circle(imageTemp, cv::Point(corange.clus[ball].centerx, corange.clus[ball].centery), 10.0, cv::Scalar(0, 0, 255), 2, 8, 0);
		writeNumber("Ally",imageTemp,golDefSup_X,golDefSup_Y,Scalar(255));
		writeNumber("Enemy",imageTemp,golAtaSup_X,golAtaSup_Y,Scalar(255));	
		cv::hconcat(imageTemp, frame, frame);

		//imwrite("/home/bakrinski/catkin_ws/src/vision/frames/imagem_fer_"+to_string(contaImagem)+".jpg", frame);
		// add(imageTemp, frame, frame);//IHC
		
		// ---------------- Mensagem ------------
		
		cout << "-- Team yellow --" << endl;
		cout << y_team.robot[0].id << endl;
		cout << ((y_team.robot[0].centery * escalaY) - campoMax_Y) << endl;
		cout << ((y_team.robot[0].centerx * escalaX) - campoMin_X) << endl;
		cout << y_team.robot[1].id << endl;
		cout << ((y_team.robot[1].centery * escalaY)- campoMax_Y) << endl;
		cout << ((y_team.robot[1].centerx * escalaX) - campoMin_X) << endl;
		cout << y_team.robot[2].id << endl;
		cout << ((y_team.robot[2].centery * escalaY)- campoMax_Y) << endl;
		cout << ((y_team.robot[2].centerx * escalaX)- campoMin_X) << endl;

		cout << "-- Team blue --" << endl;
		cout << b_team.robot[0].id << endl;
		cout << ((b_team.robot[0].centery * escalaY) - campoMin_Y) << endl;
		cout << ((b_team.robot[0].centerx * escalaX) - campoMin_X)  << endl;
		cout << b_team.robot[1].id << endl;
		cout << ((b_team.robot[1].centery * escalaY) - campoMin_Y) << endl;
		cout << ((b_team.robot[1].centerx * escalaX) - campoMin_X) << endl;
		cout << b_team.robot[2].id << endl;
		cout << ((b_team.robot[2].centery * escalaY) - campoMin_Y) << endl; 
		cout << ((b_team.robot[2].centerx * escalaX) - campoMin_X) << endl;

		if (our_team == 1){ // our team is yellow			
			// Position not found
			if ((y_team.robot[0].centerx < 0) || (y_team.robot[0].centery < 0))
			{
				// Yellow team
				msgToSituation.idr1 = y_team_posicaoAnterior.robot[0].id;
				msgToSituation.r1y = (y_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToSituation.r1x = (y_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToSituation.idr2 = y_team_posicaoAnterior.robot[1].id;    
				msgToSituation.r2y = (y_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToSituation.r2x = (y_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToSituation.idr3 = y_team_posicaoAnterior.robot[2].id;
				msgToSituation.r3y = (y_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToSituation.r3x = (y_team_posicaoAnterior.robot[2].centery * escalaY);

				msgToControl.idr1 = y_team_posicaoAnterior.robot[0].id;
				msgToControl.r1y = (y_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToControl.r1x = (y_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToControl.r1angle = y_team_posicaoAnterior.robot[0].angle;
				msgToControl.idr2 = y_team_posicaoAnterior.robot[1].id;    
				msgToControl.r2y = (y_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToControl.r2x = (y_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToControl.r2angle = y_team_posicaoAnterior.robot[1].angle;
				msgToControl.idr3 = y_team_posicaoAnterior.robot[2].id;
				msgToControl.r3y = (y_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToControl.r3x = (y_team_posicaoAnterior.robot[2].centery * escalaY);
				msgToControl.r3angle = y_team_posicaoAnterior.robot[2].angle;

				msgToStrategy.idr1 = y_team_posicaoAnterior.robot[0].id;
				msgToStrategy.r1y = (y_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToStrategy.r1x = (y_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToStrategy.r1angle = y_team_posicaoAnterior.robot[0].angle;
				msgToStrategy.idr2 = y_team_posicaoAnterior.robot[1].id;    
				msgToStrategy.r2y = (y_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToStrategy.r2x = (y_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToStrategy.r2angle = y_team_posicaoAnterior.robot[1].angle;
				msgToStrategy.idr3 = y_team_posicaoAnterior.robot[2].id;
				msgToStrategy.r3y = (y_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToStrategy.r3x = (y_team_posicaoAnterior.robot[2].centery * escalaY);
				msgToStrategy.r3angle = y_team_posicaoAnterior.robot[2].angle;

				// Blue team
				msgToSituation.ad1y = (b_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToSituation.ad1x = (b_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToSituation.ad2y = (b_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToSituation.ad2x = (b_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToSituation.ad3y = (b_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToSituation.ad3x = (b_team_posicaoAnterior.robot[2].centery * escalaY);
				
				msgToStrategy.ad1y = (b_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToStrategy.ad1x = (b_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToStrategy.ad2y = (b_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToStrategy.ad2x = (b_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToStrategy.ad3y = (b_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToStrategy.ad3x = (b_team_posicaoAnterior.robot[2].centery * escalaY);	
			}
			else{
				// Yellow team
				msgToSituation.idr1 = y_team.robot[0].id;
				msgToSituation.r1y = (y_team.robot[0].centerx * escalaX);
				msgToSituation.r1x = (y_team.robot[0].centery * escalaY);
				msgToSituation.idr2 = y_team.robot[1].id;    
				msgToSituation.r2y = (y_team.robot[1].centerx * escalaX);
				msgToSituation.r2x = (y_team.robot[1].centery * escalaY);
				msgToSituation.idr3 = y_team.robot[2].id;
				msgToSituation.r3y = (y_team.robot[2].centerx * escalaX);
				msgToSituation.r3x = (y_team.robot[2].centery * escalaY);

				msgToControl.idr1 = y_team.robot[0].id;
				msgToControl.r1y = (y_team.robot[0].centerx * escalaX);
				msgToControl.r1x = (y_team.robot[0].centery * escalaY);
				msgToControl.r1angle = y_team.robot[0].angle;
				msgToControl.idr2 = y_team.robot[1].id;    
				msgToControl.r2y = (y_team.robot[1].centerx * escalaX);
				msgToControl.r2x = (y_team.robot[1].centery * escalaY);
				msgToControl.r2angle = y_team.robot[1].angle;
				msgToControl.idr3 = y_team.robot[2].id;
				msgToControl.r3y = (y_team.robot[2].centerx * escalaX);
				msgToControl.r3x = (y_team.robot[2].centery * escalaY);
				msgToControl.r3angle = y_team.robot[2].angle;

				msgToStrategy.idr1 = y_team.robot[0].id;
				msgToStrategy.r1y = (y_team.robot[0].centerx * escalaX);
				msgToStrategy.r1x = (y_team.robot[0].centery * escalaY);
				msgToStrategy.r1angle = y_team.robot[0].angle;
				msgToStrategy.idr2 = y_team.robot[1].id;    
				msgToStrategy.r2y = (y_team.robot[1].centerx * escalaX);
				msgToStrategy.r2x = (y_team.robot[1].centery * escalaY);
				msgToStrategy.r2angle = y_team.robot[1].angle;
				msgToStrategy.idr3 = y_team.robot[2].id;
				msgToStrategy.r3y = (y_team.robot[2].centerx * escalaX);
				msgToStrategy.r3x = (y_team.robot[2].centery * escalaY);
				msgToStrategy.r3angle = y_team.robot[2].angle;			

				// Blue team
				msgToSituation.ad1y = (b_team.robot[0].centerx * escalaX);
				msgToSituation.ad1x = (b_team.robot[0].centery * escalaY);
				msgToSituation.ad2y = (b_team.robot[1].centerx * escalaX);
				msgToSituation.ad2x = (b_team.robot[1].centery * escalaY);
				msgToSituation.ad3y = (b_team.robot[2].centerx * escalaX);
				msgToSituation.ad3x = (b_team.robot[2].centery * escalaY);
				
				msgToStrategy.ad1y = (b_team.robot[0].centerx * escalaX);
				msgToStrategy.ad1x = (b_team.robot[0].centery * escalaY);
				msgToStrategy.ad2y = (b_team.robot[1].centerx * escalaX);
				msgToStrategy.ad2x = (b_team.robot[1].centery * escalaY);
				msgToStrategy.ad3y = (b_team.robot[2].centerx * escalaX);
				msgToStrategy.ad3x = (b_team.robot[2].centery * escalaY);	
			}
		}

		else if (our_team == 0){ // Our team is blue
			if ((b_team.robot[0].centerx < 0) || (b_team.robot[0].centery < 0)) // Position not found
			{
				// Blue team
				msgToSituation.idr1 = b_team_posicaoAnterior.robot[0].id;
				msgToSituation.r1y = (b_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToSituation.r1x = (b_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToSituation.idr2 = b_team_posicaoAnterior.robot[1].id;    
				msgToSituation.r2y = (b_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToSituation.r2x = (b_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToSituation.idr3 = b_team_posicaoAnterior.robot[2].id;
				msgToSituation.r3y = (b_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToSituation.r3x = (b_team_posicaoAnterior.robot[2].centery * escalaY);
				
				msgToControl.idr1 = b_team_posicaoAnterior.robot[0].id;
				msgToControl.r1y = (b_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToControl.r1x = (b_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToControl.r1angle = b_team_posicaoAnterior.robot[0].angle;
				msgToControl.idr2 = b_team_posicaoAnterior.robot[1].id;    
				msgToControl.r2y = (b_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToControl.r2x = (b_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToControl.r2angle = b_team_posicaoAnterior.robot[1].angle;
				msgToControl.idr3 = b_team_posicaoAnterior.robot[2].id;
				msgToControl.r3y = (b_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToControl.r3x = (b_team_posicaoAnterior.robot[2].centery * escalaY);
				msgToControl.r3angle = b_team_posicaoAnterior.robot[2].angle;

				msgToStrategy.idr1 = b_team_posicaoAnterior.robot[0].id;
				msgToStrategy.r1y = (b_team_posicaoAnterior.robot[0].centerx * escalaX);
				msgToStrategy.r1x = (b_team_posicaoAnterior.robot[0].centery * escalaY);
				msgToStrategy.r1angle = b_team_posicaoAnterior.robot[0].angle;
				msgToStrategy.idr2 = b_team_posicaoAnterior.robot[1].id;    
				msgToStrategy.r2y = (b_team_posicaoAnterior.robot[1].centerx * escalaX);
				msgToStrategy.r2x = (b_team_posicaoAnterior.robot[1].centery * escalaY);
				msgToStrategy.r2angle = b_team_posicaoAnterior.robot[1].angle;
				msgToStrategy.idr3 = b_team_posicaoAnterior.robot[2].id;
				msgToStrategy.r3y = (b_team_posicaoAnterior.robot[2].centerx * escalaX);
				msgToStrategy.r3x = (b_team_posicaoAnterior.robot[2].centery * escalaY);
				msgToStrategy.r3angle = b_team_posicaoAnterior.robot[2].angle;

				// Yellow team
				msgToSituation.ad1y = (y_team.robot[0].centerx * escalaX);
				msgToSituation.ad1x = (y_team.robot[0].centery * escalaY);
				msgToSituation.ad2y = (y_team.robot[1].centerx * escalaX);
				msgToSituation.ad2x = (y_team.robot[1].centery * escalaY);
				msgToSituation.ad3y = (y_team.robot[2].centerx * escalaX);
				msgToSituation.ad3x = (y_team.robot[2].centery * escalaY);
				
				msgToStrategy.ad1y = (y_team.robot[0].centerx * escalaX);
				msgToStrategy.ad1x = (y_team.robot[0].centery * escalaY);
				msgToStrategy.ad2y = (y_team.robot[1].centerx * escalaX);
				msgToStrategy.ad2x = (y_team.robot[1].centery * escalaY);
				msgToStrategy.ad3y = (y_team.robot[2].centerx * escalaX);
				msgToStrategy.ad3x = (y_team.robot[2].centery * escalaY);
			}
			else {
				msgToSituation.idr1 = b_team.robot[0].id;
				msgToSituation.r1y = (b_team.robot[0].centerx * escalaX);
				msgToSituation.r1x = (b_team.robot[0].centery * escalaY);
				msgToSituation.idr2 = b_team.robot[1].id;    
				msgToSituation.r2y = (b_team.robot[1].centerx * escalaX);
				msgToSituation.r2x = (b_team.robot[1].centery * escalaY);
				msgToSituation.idr3 = b_team.robot[2].id;
				msgToSituation.r3y = (b_team.robot[2].centerx * escalaX);
				msgToSituation.r3x = (b_team.robot[2].centery * escalaY);
				
				msgToControl.idr1 = b_team.robot[0].id;
				msgToControl.r1y = (b_team.robot[0].centerx * escalaX);
				msgToControl.r1x = (b_team.robot[0].centery * escalaY);
				msgToControl.r1angle = b_team.robot[0].angle;
				msgToControl.idr2 = b_team.robot[1].id;    
				msgToControl.r2y = (b_team.robot[1].centerx * escalaX);
				msgToControl.r2x = (b_team.robot[1].centery * escalaY);
				msgToControl.r2angle = b_team.robot[1].angle;
				msgToControl.idr3 = b_team.robot[2].id;
				msgToControl.r3y = (b_team.robot[2].centerx * escalaX);
				msgToControl.r3x = (b_team.robot[2].centery * escalaY);
				msgToControl.r3angle = b_team.robot[2].angle;

				msgToStrategy.idr1 = b_team.robot[0].id;
				msgToStrategy.r1y = (b_team.robot[0].centerx * escalaX);
				msgToStrategy.r1x = (b_team.robot[0].centery * escalaY);
				msgToStrategy.r1angle = b_team.robot[0].angle;
				msgToStrategy.idr2 = b_team.robot[1].id;    
				msgToStrategy.r2y = (b_team.robot[1].centerx * escalaX);
				msgToStrategy.r2x = (b_team.robot[1].centery * escalaY);
				msgToStrategy.r2angle = b_team.robot[1].angle;
				msgToStrategy.idr3 = b_team.robot[2].id;
				msgToStrategy.r3y = (b_team.robot[2].centerx * escalaX);
				msgToStrategy.r3x = (b_team.robot[2].centery * escalaY);
				msgToStrategy.r3angle = b_team.robot[2].angle;

				// Yellow team
				msgToSituation.ad1y = (y_team.robot[0].centerx * escalaX);
				msgToSituation.ad1x = (y_team.robot[0].centery * escalaY);
				msgToSituation.ad2y = (y_team.robot[1].centerx * escalaX);
				msgToSituation.ad2x = (y_team.robot[1].centery * escalaY);
				msgToSituation.ad3y = (y_team.robot[2].centerx * escalaX);
				msgToSituation.ad3x = (y_team.robot[2].centery * escalaY);
				
				msgToStrategy.ad1y = (y_team.robot[0].centerx * escalaX);
				msgToStrategy.ad1x = (y_team.robot[0].centery * escalaY);
				msgToStrategy.ad2y = (y_team.robot[1].centerx * escalaX);
				msgToStrategy.ad2x = (y_team.robot[1].centery * escalaY);
				msgToStrategy.ad3y = (y_team.robot[2].centerx * escalaX);
				msgToStrategy.ad3x = (y_team.robot[2].centery * escalaY);
			}			
		}
		
		msgToSituation.timestamp= (int)time(&timestamp);

		cout << "-- ball --" << endl;
		if (corange.clus[ball].size == 0) {
			cout << "-1.0" << endl;
			cout << "-1.0" << endl;
			msgToSituation.bx = -1.0;
			msgToSituation.by = -1.0;
			msgToStrategy.bx = -1.0;
			msgToStrategy.by = -1.0;
		}
		// if ( (corange.clus[0].centerx < 0) || (corange.clus[0].centery < 0)) {
		// 	cout << "-1.0" << endl;
		// 	cout << "-1.0" << endl;
		// 	msgToSituation.bx = -1.0;
		// 	msgToSituation.by = -1.0;
		// 	msgToStrategy.bx = -1.0;
		// 	msgToStrategy.by = -1.0;
		// }
		else {
			cout << (corange.clus[ball].centerx * escalaX) << endl;
			cout << (corange.clus[ball].centery * escalaY) << endl;
			msgToSituation.by = (corange.clus[ball].centerx * escalaX);
			msgToSituation.bx = (corange.clus[ball].centery * escalaY);	
			msgToStrategy.by = (corange.clus[ball].centerx * escalaX);
			msgToStrategy.bx = (corange.clus[ball].centery * escalaY);	
		}

		msgToSituation.campoMinX = campoMin_X * escalaX;
		msgToSituation.campoMinY = campoMin_Y * escalaY;
		msgToSituation.campoMaxX = campoMax_X * escalaX;
		msgToSituation.campoMaxY = campoMax_Y * escalaY;
		msgToSituation.areaDefSup_X = areaDefSup_X * escalaX;
		msgToSituation.areaDefSup_Y = areaDefSup_Y * escalaY;
		msgToSituation.areaDefInf_X = areaDefInf_X * escalaX;
		msgToSituation.areaDefInf_Y = areaDefInf_Y * escalaY;
		msgToSituation.areaAtaSup_X = areaAtaSup_X * escalaX;;
		msgToSituation.areaAtaSup_Y = areaAtaSup_Y * escalaY;
		msgToSituation.areaAtaInf_X = areaAtaInf_X * escalaX;;
		msgToSituation.areaAtaInf_Y = areaAtaInf_Y * escalaY;
		msgToSituation.golDefSup_X = golDefSup_X * escalaX;
		msgToSituation.golDefSup_Y = golDefSup_Y * escalaY;
		msgToSituation.golDefInf_X = golDefInf_X * escalaX;
		msgToSituation.golDefInf_Y = golDefInf_Y * escalaY;
		msgToSituation.golAtaSup_X = golAtaSup_X * escalaX;
		msgToSituation.golAtaSup_Y = golAtaSup_Y * escalaY;
		msgToSituation.golAtaInf_X = golAtaInf_X * escalaX;
		msgToSituation.golAtaInf_Y = golAtaInf_Y * escalaY;

		msgToStrategy.campoMinX = campoMin_X * escalaX;
		msgToStrategy.campoMinY = campoMin_Y * escalaY;
		msgToStrategy.campoMaxX = campoMax_X * escalaX;
		msgToStrategy.campoMaxY = campoMax_Y * escalaY;
		msgToStrategy.areaDefSup_X = areaDefSup_X * escalaX;
		msgToStrategy.areaDefSup_Y = areaDefSup_Y * escalaY;
		msgToStrategy.areaDefInf_X = areaDefInf_X * escalaX;
		msgToStrategy.areaDefInf_Y = areaDefInf_Y * escalaY;
		msgToStrategy.areaAtaSup_X = areaAtaSup_X * escalaX;;
		msgToStrategy.areaAtaSup_Y = areaAtaSup_Y * escalaY;
		msgToStrategy.areaAtaInf_X = areaAtaInf_X * escalaX;;
		msgToStrategy.areaAtaInf_Y = areaAtaInf_Y * escalaY;
		msgToStrategy.golDefSup_X = golDefSup_X * escalaX;
		msgToStrategy.golDefSup_Y = golDefSup_Y * escalaY;
		msgToStrategy.golDefInf_X = golDefInf_X * escalaX;
		msgToStrategy.golDefInf_Y = golDefInf_Y * escalaY;
		msgToStrategy.golAtaSup_X = golAtaSup_X * escalaX;
		msgToStrategy.golAtaSup_Y = golAtaSup_Y * escalaY;
		msgToStrategy.golAtaInf_X = golAtaInf_X * escalaX;
		msgToStrategy.golAtaInf_Y = golAtaInf_Y * escalaY;


		msgToAnalysis.A = 4;
		msgToAnalysis.B = 5;
		msgToAnalysis.C = 6;
		pubToSituation.publish(msgToSituation);
		ros::spinOnce();

		pubToAnalysis.publish(msgToAnalysis);
		ros::spinOnce();

		pubToStrategy.publish(msgToStrategy);
		ros::spinOnce();

		pubToControl.publish(msgToControl);
		ros::spinOnce();

		loop_rate.sleep();

		//Guarda a posicao anterior
		// copiaEstadoTime(&y_team_posicaoAnterior, &y_team);
		// copiaEstadoTime(&b_team_posicaoAnterior, &b_team);
		// zeraTime(&y_team);
		// zeraTime(&b_team);

		contaImagem++;
		imshow("edges", frame);//IHC

		if(waitKey(30) >= 0) continue;

	    	/*std::cout << "Enter with three numbers:" << std::endl;
	    	std::cin >> a;
	    	std::cin >> b;
	    	std::cin >> c;

	    	if(a == -1 && b == -1 && c == -1)
	      	exit(0);
	    	*/
	    	/*msgToSituation.A = 1;
	    	msgToSituation.B = 2;
	    	msgToSituation.C = 3;*/
	}

free(cyellow.clus);
free(cblue.clus);
free(cred.clus);
  //cap.release();
}
