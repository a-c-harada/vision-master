#include <stdlib.h>
#include <unistd.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

#define EXPECTED_CLUSTERS 16
#define MIN_CLUSTER_SIZE 25
#define MAX_CLUSTER_SIZE 500
#define TEAM_SIZE 3

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

void writeNumber(string text, Mat img, int x, int y, cv::Scalar color)
{
  int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1;
  int thickness = 2;

  int baseline=0;
  Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
  baseline += thickness;

  // center the text
  // Point textOrg((img.cols - textSize.width)/2,
  //               (img.rows + textSize.height)/2);
  Point textOrg(x- textSize.width,y- textSize.height);

  // // draw the box
  // rectangle(img, textOrg + Point(0, baseline),
  //           textOrg + Point(textSize.width, -textSize.height),
  //           Scalar(0,0,255));
  // // ... and the baseline first
  // line(img, textOrg + Point(0, thickness),
  //      textOrg + Point(textSize.width, thickness),
  //      Scalar(0, 0, 255));

  // then put the text itself
  putText(img, text, textOrg, fontFace, fontScale, color, thickness, 8);
}

// void r_add(team& t, relation& rel)
// {
//   if(t.size<TEAM_SIZE)
//   {
//     t.size++;
//     t.r[t.size].uni_i=rel.i;
//     t.r[t.size].uni_j=rel.j;
//   }
// }

// void r_show(team& t)
// {
//   for (int i = 0; i < t.size; i++) {
//     cout<<"robot "<< i << " i=" << t.r[i].uni_i << " j=" << t.r[i].uni_j<<endl;
//   }
//
// }


float EuclideanDistance(float xa, float ya, float xb, float yb){
  return sqrt((xa-xb)*(xa-xb)+(ya-yb)*(ya-yb));
}

// void findNear(cluster_vec& a,cluster_vec& b)
// {
//   float distance;
//   relation r;
//   r.min=5000;
//   r.i=0;
//   r.j=0;
//   r.angle=0;
//   for (int i = 0; i < a.len; i++)
//   {
//     if(a.clus[i].size>0)
//     {
//       for (int j = 0; j < b.len; j++)
//       {
//         if(b.clus[j].size>0)
//         {
//           distance = EuclideanDistance(a.clus[i].centerx,a.clus[i].centery,b.clus[j].centerx,b.clus[j].centery);
//           if(distance<r.min)
//           {
//             r.min=distance;
//             r.i=i;
//             r.j=j;
//           }
//         }
//       }
//     }
//   }
//
//
//   if( (b.clus[r.j].centerx-a.clus[r.i].centerx)<-(r.min*0.66) ){
//     r.angle+=270;
//   }
//   else
//   {
//     if( (b.clus[r.j].centerx-a.clus[r.i].centerx)>(r.min*0.66) ){
//       r.angle+=90;
//     }
//     if( (b.clus[r.j].centery-a.clus[r.i].centery)<-(r.min*0.66) ){
//       r.angle+=180;
//     }
//   }
//
//   float temp= ( ((UNIFORM_HEIGHT*a.clus[r.i].diffx)/UNIFORM_WIDTH) - a.clus[r.i].diffy) /
//               ( ((UNIFORM_HEIGHT*UNIFORM_HEIGHT)/UNIFORM_WIDTH) - UNIFORM_WIDTH );
//
//
//   cout<<"temp=" << temp;
//   if (temp>1.0f) {
//     temp=1.0f;
//   }
//   cout<<"\nasin(temp)=" << asin(temp)*(180.0/3.14);
//   // r.angle+=asin(temp)*(180.0/3.14);
//
//   cout<<" angle=" << r.angle;
//   cout << " i="<<r.i<<" j="<<r.j<<" min="<<r.min << endl;
//
// }


team findNear(cluster_vec& a,cluster_vec& b)
{
  team test;
  test.size=0;

  float distance;
  relation r;
  r.min=MAX_CLUSTER_SIZE;
  r.i=0;
  r.j=0;


  for (int i = 0; i < a.len; i++)
  {
    if(a.clus[i].size>0)
    {
      for (int j = i+1; j < a.len; j++)
      {
        if(a.clus[j].size>0)
        {
          distance = EuclideanDistance(a.clus[i].centerx,a.clus[i].centery,a.clus[j].centerx,a.clus[j].centery);
          if( distance<r.min )//need to check min distance defined
          {
            r.min=distance;
            r.i=i;
            r.j=j;
            // r_add(t, r);

            if(test.size<TEAM_SIZE)
            {
              test.robot[test.size].uni_i=r.i;
              test.robot[test.size].uni_j=r.j;
              test.robot[test.size].min_ij=r.min;
              test.size++;
            }
          }
        }
      }
    }
  }//endfor

  for (int i = 0; i < test.size; i++)
  {
    for (int j = 0; j < b.len; j++)
    {
      if(b.clus[j].size>0)
      {
        distance = EuclideanDistance(a.clus[test.robot[i].uni_i].centerx,a.clus[test.robot[i].uni_i].centery,b.clus[j].centerx,b.clus[j].centery);
        if(distance<r.min)
        {
          r.min=distance;
          r.i=i;
          r.j=j;
        }
      }
    }//endfor
    test.robot[i].red_k=r.j;
    r.i=test.robot[i].uni_i;
    test.robot[i].angle=0;
    float mycos=0;

    float leftx=a.clus[test.robot[i].uni_j].centerx;
    float rightx=a.clus[test.robot[i].uni_i].centerx;
    float bottomy=a.clus[test.robot[i].uni_j].centery;
    float uppery=a.clus[test.robot[i].uni_i].centery;
    if( a.clus[test.robot[i].uni_i].centerx < a.clus[test.robot[i].uni_j].centerx)
    {
      leftx=a.clus[test.robot[i].uni_i].centerx;
      rightx=a.clus[test.robot[i].uni_j].centerx;
    }
    if( a.clus[test.robot[i].uni_i].centery > a.clus[test.robot[i].uni_j].centery)
    {
      bottomy=a.clus[test.robot[i].uni_i].centery;
      uppery=a.clus[test.robot[i].uni_j].centery;
    }

    // cout<< " leftx="<<leftx<<endl;
    // cout<< " rightx="<<rightx<<endl;
    // cout<< " bottomy="<<bottomy<<endl;
    // cout<< " uppery="<<uppery<<endl;
    test.robot[i].centerx= leftx+((rightx-leftx)/2);
    test.robot[i].centery= uppery+((bottomy-uppery)/2);


    // if( (b.clus[r.j].centerx-a.clus[r.i].centerx)<-(r.min*0.66) ){
    if( (b.clus[r.j].centerx-test.robot[i].centerx)<-10.0f )
    {
      // mycos=(rightx-leftx)/test.robot[i].min_ij;
      // test.robot[i].angle=acos(mycos)*(180.0/3.14);

      // mycos=(bottomy-uppery)/test.robot[i].min_ij;
      // mycos=(uppery-bottomy)/test.robot[i].min_ij;
      // test.robot[i].angle=asin(mycos)*(180.0/3.14);

      test.robot[i].angle+=270;
    }
    else
    {
      if( (b.clus[r.j].centerx-test.robot[i].centerx)>10.0f  )
      {
        // mycos=(rightx-leftx)/test.robot[i].min_ij;
        // test.robot[i].angle=acos(mycos)*(180.0/3.14);
        test.robot[i].angle+=90;
      }
      else
      {
        if( (b.clus[r.j].centery-test.robot[i].centery)<-10.0f  )
        {
          // mycos=(rightx-leftx)/test.robot[i].min_ij;
          // test.robot[i].angle=acos(mycos)*(180.0/3.14);
          test.robot[i].angle+=180;
        }
        else
        {
          //0degree
          mycos=(rightx-leftx)/test.robot[i].min_ij;
          test.robot[i].angle=acos(mycos)*(180.0/3.14);
        }
      }
    }//endelse

    cout << "robot "<<i
    <<" ti="<<test.robot[i].uni_i
    <<" tj="<< test.robot[i].uni_j
    <<" tk=" << test.robot[i].red_k
    <<" k_size=" << b.clus[test.robot[i].red_k].size
    <<" angle="<<test.robot[i].angle
    <<" centerx="<<test.robot[i].centerx
    <<" centery="<<test.robot[i].centery<< endl;
  }//endfor

  // cout << " i="<<r.i<<" j="<<r.j<<" min="<<r.min << endl;

  return test;
}//endfunc

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
// void check_neighbor(uint8_t* img,int i,int j,int n,int m, uint8_t* checked, cluster* c)
{
  if(c->size > MAX_CLUSTER_SIZE)
  {
    return;
  }

  if( checked[(i*m)+j]==0 )
  {
    float avg=(img[(i*m*3+j*3)]+img[(i*m*3+j*3)+1]+img[(i*m*3+j*3)+2]);
    float r=(float)img[(i*m*3+j*3)+2]/avg;
    float g=(float)img[(i*m*3+j*3)+1]/avg;
    float b=(float)img[(i*m*3+j*3)]/avg;
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

void threshold(Mat& image,colors* obj,Mat& frame, cluster_vec& c)
{

  uint8_t *img = image.data;
  int n=image.rows;
  int m=image.cols;

  uint8_t* checked=(uint8_t*)calloc(n*m,sizeof(uint8_t) );

  int counter=0;
  int maxCluster=c.len;
  for (int i = 0; i < maxCluster; i++)
  {
    init_cluster(&c.clus[i],m,0,n,0,0);
  }


  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
    {
      float avg=(img[(i*m*3+j*3)]+img[(i*m*3+j*3)+1]+img[(i*m*3+j*3)+2]);
      float r=(float)img[(i*m*3+j*3)+2]/avg;
      float g=(float)img[(i*m*3+j*3)+1]/avg;
      float b=(float)img[(i*m*3+j*3)]/avg;
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
              // counter=0;
              // for (int i = 0; i < maxCluster; i++)
              // {
              //   init_cluster(&c.clus[i],m,0,n,0,0);
              // }
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
      // else
      // {
      //   img[(i*m*3+j*3)+0]=0;
      //   img[(i*m*3+j*3)+1]=0;
      //   img[(i*m*3+j*3)+2]=0;
      // }
    }
  }

  for (int k = 0; k < counter; k++)
  {
    // if(c.clus[k].size>0)
    // {
      c.clus[k].diffx=(c.clus[k].xmax-c.clus[k].xmin);
      c.clus[k].diffy=(c.clus[k].ymax-c.clus[k].ymin);
      c.clus[k].centerx=c.clus[k].xmin+(c.clus[k].diffx/2);
      c.clus[k].centery=c.clus[k].ymin+(c.clus[k].diffy/2);
      cout <<"cluster["<<k<<"] size="<< c.clus[k].size;
      cout <<" centerx="<< c.clus[k].centerx;
      cout <<" centery="<< c.clus[k].centery;
      cout <<" diffx="<< c.clus[k].diffx;
      cout <<" diffy="<< c.clus[k].diffy<< endl;
      // cout <<" xmin="<< c.clus[k].xmin;
      // cout <<" xmax="<< c.clus[k].xmax;
      // cout <<" ymin="<< c.clus[k].ymin;
      // cout <<" ymax="<< c.clus[k].ymax<< endl;

      rectangle(
        frame,
        cv::Point(c.clus[k].xmin, c.clus[k].ymin),
        cv::Point(c.clus[k].xmax, c.clus[k].ymax),
        cv::Scalar(obj->highB*255, obj->highG*255, obj->highR*255)
      );
    // }

  }

  c.len=maxCluster;
  free(checked);
}

int main( int argc, char** argv )
{
  Mat img_anterior, img_atual, antImage_c, currImage_c;

  //VideoCapture cap("../video.mp4"); // open the default camera
  //if(!cap.isOpened())  // check if we succeeded
  //  return -1;
  //      cap >> antImage_c; // get a new frame from camera
        
  Mat cameraMatrix = (Mat_<double>(3,3) << 663.12266191,0.,298.54197567,0.,664.55319541,271.03745091,0.,0.,1.);
  
  Mat distCoeffs = (Mat_<double>(5,1) << -0.43831627,0.29401438,-0.00404992,-0.00459204,-0.12562864);

  Mat map1, map2;
  Rect roi;
  Mat NewCameraMtx = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size(640, 480), 1, Size(640, 480), &roi);

  initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), NewCameraMtx, Size(640, 480), CV_16SC2, map1, map2);




  VideoCapture cap(argv[1]); // open the default camera
  //cout << to_string(argv[1]);
  //return -1;

  //VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  // cap.set(39, atoi(argv[1]));

  Mat image;
  Mat frame(image.rows, image.cols, CV_8UC3, Scalar(0));

  colors yellow;
  init_color(&yellow,0.34f,1.0f,0.33f,1.0f,0.0f,0.32f);

  colors blue;
  init_color(&blue,0.15f,0.35f,0.15f,0.35f,0.45f,1.0f);
  // init_color(&blue,0.0f,0.4f,0.0f,0.4f,0.5f,1.0f);

  colors red;
  init_color(&red,0.4f,1.0f,0.1f,0.35f,0.1f,0.35f);

  cluster_vec cyellow;
  cyellow.len=EXPECTED_CLUSTERS;
  cyellow.clus=(cluster*)malloc(cyellow.len*sizeof(cluster));

  cluster_vec cblue;
  cblue.len=EXPECTED_CLUSTERS;
  cblue.clus=(cluster*)malloc(cblue.len*sizeof(cluster));

  cluster_vec cred;
  cred.len=EXPECTED_CLUSTERS;
  cred.clus=(cluster*)malloc(cred.len*sizeof(cluster));

  team y_team;
  y_team.size=-1;
  team b_team;
  b_team.size=-1;
  for(;;)
  {
    // y_team.size=-1;
    // b_team.size=-1;
    cap >> image;

    // GaussianBlur(image, image, Size(15,15),0,0 );
    // image = fastNlMeansDenoisingColored(image,None,10,10,7,21);

    frame = Mat::zeros(image.rows, image.cols, CV_8UC3);
  

    cout<<"yellow"<<endl;
    threshold(image,&yellow,frame,cyellow);

    cout<<"blue"<<endl;
    threshold(image,&blue,frame,cblue);

    cout<<"red"<<endl;
    threshold(image,&red,frame,cred);

    cout<<"yellow"<<endl;
    y_team=findNear(cyellow,cred);
    // r_show(y_team);
    cout<<"blue"<<endl;
    b_team=findNear(cblue,cred);


    add(image,frame,frame);

    for(int i=0;i<y_team.size;i++){
      writeNumber(to_string(i),frame,y_team.robot[i].centerx,y_team.robot[i].centery,cv::Scalar(0,255,255) );
    }
    for(int i=0;i<b_team.size;i++){
      writeNumber(to_string(i),frame,b_team.robot[i].centerx,b_team.robot[i].centery,cv::Scalar(255,0,0) );
    }


    cout<<endl<<endl;

    remap(frame, frame, map1, map2, INTER_LINEAR);
    frame = frame(roi);
    
    imshow("edges", frame);
    if(waitKey(30) >= 0) break;
  }
  imwrite("temp.jpg",frame);

  free(cyellow.clus);
  free(cblue.clus);
  free(cred.clus);
  cap.release();

  return 0;
}
