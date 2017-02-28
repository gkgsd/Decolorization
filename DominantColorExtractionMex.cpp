#include<mex.h>
#include <vector>
#include <cmath>
using namespace std;


void rgbtolab(int* rin, int* gin, int* bin, int sz, double* lvec, double* avec, double* bvec)
{
    int i; int sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    const double epsilon = 0.008856;	//actual CIE standard
    const double kappa   = 903.3;		//actual CIE standard
    
    const double Xr = 0.950456;	//reference white
    const double Yr = 1.0;		//reference white
    const double Zr = 1.088754;	//reference white
    double xr,yr,zr;
    double fx, fy, fz;
    double lval,aval,bval;
    
    for(i = 0; i < sz; i++)
    {
        sR = rin[i]; sG = gin[i]; sB = bin[i];
        R = sR/255.0;
        G = sG/255.0;
        B = sB/255.0;
        
        if(R <= 0.04045)	r = R/12.92;
        else				r = pow((R+0.055)/1.055,2.4);
        if(G <= 0.04045)	g = G/12.92;
        else				g = pow((G+0.055)/1.055,2.4);
        if(B <= 0.04045)	b = B/12.92;
        else				b = pow((B+0.055)/1.055,2.4);
        
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        
        //------------------------
        // XYZ to LAB conversion
        //------------------------
        xr = X/Xr;
        yr = Y/Yr;
        zr = Z/Zr;
        
        if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        else				fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        else				fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        else				fz = (kappa*zr + 16.0)/116.0;
        
        lval = 116.0*fy-16.0;
        aval = 500.0*(fx-fy);
        bval = 200.0*(fy-fz);
        
        lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
    }
}

void ConnectSpatially(int* labels, int width, int height,int* nlabels, int* finalNumberOfLabels, int minsz)
{
    int i,j,k;
    int n,c,count;
    int x,y;
    int ind;
    int oindex, adjlabel;
    int label;
    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};
    const int sz = width*height;
    int* xvec = (int*)mxMalloc(sizeof(int)*sz);
    int* yvec = (int*)mxMalloc(sizeof(int)*sz);
    
    for( i = 0; i < sz; i++ ) nlabels[i] = -1;
    oindex = 0;
    adjlabel = 0;//adjacent label
    label = 0;
    for( j = 0; j < height; j++ )
    {
        for( k = 0; k < width; k++ )
        {
            if( 0 > nlabels[oindex] )
            {
                nlabels[oindex] = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                {for( n = 0; n < 4; n++ )
                {
                    int x = xvec[0] + dx4[n];
                    int y = yvec[0] + dy4[n];
                    if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                    {
                        int nindex = y*width + x;
                        if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
                    }
                }}
                
                count = 1;
                for( c = 0; c < count; c++ )
                {
                    for( n = 0; n < 4; n++ )
                    {
                        x = xvec[c] + dx4[n];
                        y = yvec[c] + dy4[n];
                        
                        if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                        {
                            int nindex = y*width + x;
                            
                            if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }
                        
                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if(count <= minsz)
                {
                    for( c = 0; c < count; c++ )
                    {
                        ind = yvec[c]*width+xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    *finalNumberOfLabels = label;
    
    mxFree(xvec);
    mxFree(yvec);
}

/*

//===========================================================================
///	GetKValues_LAB
///
/// The k seed values are automatically determined as the peaks of RGB
///	histogram within a chosen search window.
///
/// Ths version of the function uses pointers
//===========================================================================
void DominantColorExtraction::GetKValues_LAB(
        double*        lvec,
        double*		avec,
        double*		bvec,
        const int          width,
        const int          height,
        double*     		kseedsl,
        double*     		kseedsa,
        double*     		kseedsb,
        int*                numseeds)
{
    const int	BINS	= 10;//need not be a power of 2 (unlike in the case of RGB histogram case)
    int*** labhist = new int**[BINS];
    bool*** seen = new bool**[BINS];
    int s1;
    for(s1 = 0; s1 < BINS; s1++ )
    {
        labhist[s1] = new int*[BINS];
        seen[s1] = new bool*[BINS];
        int s2;
        for( s2 = 0; s2 < BINS; s2++ )
        {
            labhist[s1][s2] = new int[BINS];
            seen[s1][s2] = new bool[BINS];
        }
    }
    {
    int i,j,k;
    for(i = 0; i < BINS; i++) for(j = 0; j < BINS; j++) for(k = 0; k < BINS; k++){labhist[i][j][k] = 0; seen[i][j][k] = false;}
    }

    //------------------------
    // Create lab histogram
    //------------------------
    double maxl(-1), maxa(-1), maxb(-1);
    double minl(1<<30), mina(1<<30), minb(1<<30);

    int sz = width*height;
    int j;
    for(j = 0; j < sz; j++ )
    {
        if( lvec[j] < minl )minl = lvec[j];
        if( avec[j] < mina )mina = avec[j];
        if( bvec[j] < minb )minb = bvec[j];
        if( lvec[j] > maxl )maxl = lvec[j];
        if( avec[j] > maxa )maxa = avec[j];
        if( bvec[j] > maxb )maxb = bvec[j];
    }

    double rangel = maxl-minl;
    double rangea = maxa-mina;
    double rangeb = maxb-minb;
    for( j = 0; j < sz; j++ )
    {
        int lindex = int( 0.5f + (BINS-1)*(lvec[j] - minl)/rangel );
        int aindex = int( 0.5f + (BINS-1)*(avec[j] - mina)/rangea );
        int bindex = int( 0.5f + (BINS-1)*(bvec[j] - minb)/rangeb );

        labhist[lindex][aindex][bindex]++;
    }

    //-----------------------------------------------------------------
    // Now that we have the histogram, we need to find the peaks in it.
    // Using the hill-climbing algorithm for this
    //-----------------------------------------------------------------
    const int MAXPEAKS = 30;
    double lvals[MAXPEAKS],avals[MAXPEAKS],bvals[MAXPEAKS];int peakcount = 0;
    const int w = 1;// window size
    int l;
    for( l = 0; l < BINS; l++ )
    {
        int a;
        for( a = 0; a < BINS; a++ )
        {
            int b;
            for( b = 0; b < BINS; b++ )
            {
                bool islargest(true);
                //-----------------------------------------------------------------
                // Look around in the neighbourhood. If the current bin is the
                // largest in the neighbourhood, it is taken to be a k
                //-----------------------------------------------------------------
                int x;
                for( x = l-w; x <= l+w; x++ )
                {
                    int y;
                    for( y = a-w; y <= a+w; y++ )
                    {
                        int z;
                        for( z = b-w; z <= b+w; z++ )
                        {
                            if( (x >= 0 && x < BINS) && (y >= 0 && y < BINS)  && (z >= 0 && z < BINS) && !(x==l && y==a && z==b) )
                            {
                                if( labhist[l][a][b] <= labhist[x][y][z] )
                                {
                                    islargest  = false;
                                }
                            }
                        }
                    }
                }

                //-----------------------------------------------------------------
                // if curval has not changed, then it is a peak value, so store it
                // as a k value (second condition is for avoiding the case of two
                ///equal bins)
                //-----------------------------------------------------------------
                if( true == islargest && peakcount < MAXPEAKS)
                {
                    lvals[peakcount] = int( 0.5f + minl + l*rangel/double(BINS-1) );
                    avals[peakcount] = int( 0.5f + mina + a*rangea/double(BINS-1) );
                    bvals[peakcount] = int( 0.5f + minb + b*rangeb/double(BINS-1) );

                    peakcount++;
                }
            }
        }
    }
    if( peakcount < 2 )
    {
        kseedsl = new double[1];kseedsl[0] = lvec[0];
        kseedsa = new double[1];kseedsa[0] = avec[0];
        kseedsb = new double[1];kseedsb[0] = bvec[0];
        numseeds = 1;
    }
    else
    {
        kseedsl = new double[peakcount];
        kseedsa = new double[peakcount];
        kseedsb = new double[peakcount];
        for(int n = 0; n < peakcount; n++)
        {
            kseedsl[n] = lvals[n];
            kseedsa[n] = avals[n];
            kseedsb[n] = bvals[n];
        }
        *numseeds = peakcount;
    }

    for( s1 = 0; s1 < BINS; s1++ )
    {
        int s2;
        for( s2 = 0; s2 < BINS; s2++ )
        {
            delete [] labhist[s1][s2];
            delete [] seen[s1][s2];
        }
        delete [] labhist[s1];
        delete [] seen[s1];
    }
    delete [] labhist;
    delete [] seen;
}

//===========================================================================
///	GetDominantColors
///
///	Performs k mean segmentation. The Hill Climbing algorithm is used to find
///	k local maxima in the RGB histogram, and uses that many seeds.
///
/// This version of the function uses pointers.
//===========================================================================
void DominantColorExtraction::GetDominantColors(
        unsigned int*	img,
        double*		lvec,
        double*		avec,
        double*		bvec,
        const int			width,
        const int			height,
        int*                klabels
        int*				numlabels)//,
//        unsigned int*      avgcolors,
//        unsigned int*      closestcolors,
//        int*               clustersizes)
        //vector<unsigned int>& domcolorsimage)
{
    int i, j, k;
    double* kseedsl = 0; double* kseedsa = 0; double* kseedsb = 0;
    int sz = width*height;
    
    if(1)//choose seeds on the lab histogram
    {
        GetKValues_LAB(lvec, avec, bvec, width, height, kseedsl, kseedsa, kseedsb, numlabels);
    }

    const int numk = *numlabels;
    float cumerr(99999.9f);
    int numitr(0);
    int* closestpixelindex = new int[numk];
//    double* sigmal = new double[numk];memset(sigmal,0,sizeof(double)*numk);
//    double* sigmaa = new double[numk];memset(sigmaa,0,sizeof(double)*numk);
//    double* sigmab = new double[numk];memset(sigmab,0,sizeof(double)*numk);
//    double* invsz = new double[numk];
    
    double* sigmal    = mxMalloc( sizeof(double)      * numk ) ;
    double* sigmaa    = mxMalloc( sizeof(double)      * numk ) ;
    double* sigmab    = mxMalloc( sizeof(double)      * numk ) ;
    double* mindistvec= mxMalloc( sizeof(double)      * sz ) ;
    
    for(k = 0; k < numk; k++)
    {
        sigmal[k] = 0; sigmaa[k] = 0; sigmab[k] = 0;
    }
    for(i = 0; i < sz; i++)

//    avgcolors = new unsigned int[numk];
//    closestcolors = new unsigned int[numk];
//    clustersizes = new int[numk];

    const double temp = pow(2.0, 30.0);
    vector<double> mindistvec(sz,temp);

    while( fabs(cumerr) > 0.5 && numitr < 20 )
    {
        //------
        cumerr = 0;
        numitr++;
        //------
        mindistvec.assign(sz,temp);
        for( int k = 0; k < numk;k++ )
        {
            for( int i = 0; i < sz; i++ )
            {
                double dist =	(lvec[i] - kseedsl[k])*(lvec[i] - kseedsl[k]) +
                                (avec[i] - kseedsa[k])*(avec[i] - kseedsa[k]) +
                                (bvec[i] - kseedsb[k])*(bvec[i] - kseedsb[k]);

                if( dist < mindistvec[i] )
                {
                    mindistvec[i] = dist;
                    klabels[i] = k;
                }
            }
        }

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        memset(sigmal,0,sizeof(double)*numk);
        memset(sigmaa,0,sizeof(double)*numk);
        memset(sigmab,0,sizeof(double)*numk);
        memset(clustersizes,0,sizeof(int)*numk);

        int j;
        for( j = 0; j < sz; j++ )
        {
            sigmal[klabels[j]] += lvec[j];
            sigmaa[klabels[j]] += avec[j];
            sigmab[klabels[j]] += bvec[j];

            clustersizes[klabels[j]]++;
        }
        int k;
        {for( k = 0; k < numk; k++ )
        {
            invsz[k] = 1.0/max(1, clustersizes[k]);
        }}
        {for( k = 0; k < numk; k++ )
        {
            cumerr += (kseedsl[k] - sigmal[k]*invsz[k]);
            cumerr += (kseedsa[k] - sigmaa[k]*invsz[k]);
            cumerr += (kseedsb[k] - sigmab[k]*invsz[k]);

            kseedsl[k] = sigmal[k]*invsz[k];
            kseedsa[k] = sigmaa[k]*invsz[k];
            kseedsb[k] = sigmab[k]*invsz[k];
        }}
    }
    
    
    delete [] kseedsl; delete [] kseedsa; delete [] kseedsb;
    delete [] invsz;
    delete [] sigmal; delete [] sigmaa; delete [] sigmab;
    delete [] ravg; delete [] gavg; delete [] bavg; delete [] segsz;
    delete [] pixcolvec;
}*/



//===========================================================================
///	ClusterWithKnownDistance
///
/// Works very well with distance 1000. No need to do a random shuffle.
/// Results seem steady even with random shuffle, which is good.
/// 13 December 2014
//===========================================================================
void ClusterWithKnownDistance(//unsigned int*	img,
                                                       double*		lvec,
                                                       double*		avec,
                                                       double*		bvec,
                                                       const int					width,
                                                       const int					height,
                                                       int*				klabels,
                                                       int*						numlabels,
                                                       const double&               maxclusterdistance)
{
    int sz = width*height;
    
    vector<double> kl(0), ka(0), kb(0);
    vector<double> lsum(0), asum(0), bsum(0);
    vector<int> ksz(0);
    //klabels.resize(sz,-1);
    int i = 0;//indvec[0];
    kl.push_back(lvec[i]);ka.push_back(avec[i]);kb.push_back(bvec[i]);
    lsum.push_back(lvec[i]);asum.push_back(avec[i]);bsum.push_back(bvec[i]);
    ksz.push_back(1);
    klabels[i] = 0;
    int numk = 1;
    const double DISTTHRESH = maxclusterdistance*maxclusterdistance;
    

    for(int ind = 1; ind < sz; ind++)
    {
        i = ind;//indvec[ind];
        int bestk = -1; double bestdist = DBL_MAX;
        for(int k = 0; k < numk; k++)
        {
            double dist = (kl[k]-lvec[i])*(kl[k]-lvec[i])+(ka[k]-avec[i])*(ka[k]-avec[i])+(kb[k]-bvec[i])*(kb[k]-bvec[i]);
            if(dist < DISTTHRESH)
            {
                if(dist < bestdist)
                {
                    bestdist = dist;
                    bestk = k;
                    klabels[i] = k;
                }
            }
        }
        if(bestk < 0)//assign a new centroid
        {
            kl.push_back(lvec[i]);ka.push_back(avec[i]);kb.push_back(bvec[i]);
            lsum.push_back(lvec[i]);asum.push_back(avec[i]);bsum.push_back(bvec[i]);
            ksz.push_back(1);
            klabels[i] = numk;
            numk++;
        }
        else
        {
            lsum[bestk] += lvec[i];asum[bestk] += avec[i];bsum[bestk] += bvec[i];ksz[bestk]++;
            kl[bestk] = lsum[bestk]/ksz[bestk];ka[bestk] = asum[bestk]/ksz[bestk];kb[bestk] = bsum[bestk]/ksz[bestk];
        }
    }

    
    for(int itr = 0; itr < 10; itr++)
    {
        lsum.assign(numk,0);asum.assign(numk,0);bsum.assign(numk,0);ksz.assign(numk,0);
        for(int i = 0; i < sz; i++)//now refine the clusters
        {
            int bestk = -1; double bestdist = DBL_MAX;
            for(int k = 0; k < numk; k++)
            {
                double dist = (kl[k]-lvec[i])*(kl[k]-lvec[i])+(ka[k]-avec[i])*(ka[k]-avec[i])+(kb[k]-bvec[i])*(kb[k]-bvec[i]);
                if(dist < bestdist)
                {
                    bestdist = dist;
                    bestk = k;
                    klabels[i] = k;
                }
            }
            lsum[bestk] += lvec[i];asum[bestk] += avec[i];bsum[bestk] += bvec[i];ksz[bestk]++;
        }
        for(int k = 0; k < numk; k++)
        {
            if(ksz[k] > 0)
            {
                kl[k] = lsum[k]/ksz[k];ka[k] = asum[k]/ksz[k];kb[k] = bsum[k]/ksz[k];
            }
        }
    }

    
    int numnonzero(0);for(int k = 0; k < numk; k++) if(ksz[k] > 0) numnonzero++;
    vector<double> nkl(numnonzero),nka(numnonzero),nkb(numnonzero);
    vector<int> nksz(numnonzero);
    for(int k = 0, n = 0; k < numk; k++)
    {
        if(ksz[k] > 0)
        {
            nkl[n] = kl[k]; nka[n] = ka[k]; nkb[n] = kb[k]; nksz[n] = ksz[k]; n++;
        }
    }
    kl = nkl; ka = nka; kb = nkb; ksz = nksz;//do the swaps
    
    *numlabels = numk;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int width;
    int height;
    int sz;
    int i, ii;
    int x, y;
    int* rin; int* gin; int* bin;
    int* klabels;
    int* clabels;
    int* outklabels;
    int* outclabels;
    int numklabels;
    int numclabels;
    int* numoutklabels;
    int* numoutclabels;
    double* lvec; double* avec; double* bvec;
    unsigned int* img;
    int step;
    int* seedIndices;
    int numseeds;
    double* kseedsx;double* kseedsy;
    double* kseedsl;double* kseedsa;double* kseedsb;
    int k;
    int minsz;
    const mwSize* dims;//int* dims;
    mwSize numdims;
    
    unsigned char* imgbytes;
    int numelements;
    int distanceThreshold;

//    
//    if (nrhs < 1) {
//        mexErrMsgTxt("At least one argument is required.") ;
//    } else if(nrhs > 3) {
//        mexErrMsgTxt("Too many input arguments.");
//    }
//    if(nlhs!=4) {
//        mexErrMsgIdAndTxt("SLIC:nlhs","Two outputs required, a labels and the number of labels, i.e superpixels.");
//    }
    //---------------------------
    numelements   = mxGetNumberOfElements(prhs[0]) ;
    numdims = mxGetNumberOfDimensions(prhs[0]);
    dims  = mxGetDimensions(prhs[0]) ;
    imgbytes  = (unsigned char*)mxGetData(prhs[0]) ;//mxGetData returns a void pointer, so cast it
    width = dims[1]; height = dims[0];//Note: first dimension provided is height and second is width
    sz = width*height;
    //---------------------------
    distanceThreshold  = mxGetScalar(prhs[1]);
    minsz = mxGetScalar(prhs[2]);
    //compactness     = mxGetScalar(prhs[2]);
    
    //---------------------------
    // Allocate memory
    //---------------------------
    rin    = (int*)mxMalloc( sizeof(int)      * sz ) ;
    gin    = (int*)mxMalloc( sizeof(int)      * sz ) ;
    bin    = (int*)mxMalloc( sizeof(int)      * sz ) ;
    lvec    = (double*)mxMalloc( sizeof(double)      * sz ) ;
    avec    = (double*)mxMalloc( sizeof(double)      * sz ) ;
    bvec    = (double*)mxMalloc( sizeof(double)      * sz ) ;
    img     = (unsigned int*)mxMalloc( sizeof(unsigned int)* sz );
    klabels = (int*)mxMalloc( sizeof(int)         * sz );//original k-means color labels
    clabels = (int*)mxMalloc( sizeof(int)         * sz );//spatial labels
    seedIndices = (int*)mxMalloc( sizeof(int)     * sz );
    
    //---------------------------
    // Perform color conversion
    //---------------------------
    //if(2 == numdims)
    if(numelements/sz == 1)//if it is a grayscale image, copy the values directly into the lab vectors
    {
        for(x = 0, ii = 0; x < width; x++)//reading data from column-major MATLAB matrics to row-major C matrices (i.e perform transpose)
        {
            for(y = 0; y < height; y++)
            {
                i = y*width+x;
                lvec[i] = imgbytes[ii];
                avec[i] = imgbytes[ii];
                bvec[i] = imgbytes[ii];
                ii++;
            }
        }
    }
    else//else covert from rgb to lab
    {
        for(x = 0, ii = 0; x < width; x++)//reading data from column-major MATLAB matrics to row-major C matrices (i.e perform transpose)
        {
            for(y = 0; y < height; y++)
            {
                i = y*width+x;
                rin[i] = imgbytes[ii];
                gin[i] = imgbytes[ii+sz];
                bin[i] = imgbytes[ii+sz+sz];
                img[i] = rin[i] << 16 | gin[i] << 8 | bin[i];
                ii++;
            }
        }
        rgbtolab(rin,gin,bin,sz,lvec,avec,bvec);
    }
    
    
//    kseedsx    = (double*)mxMalloc( sizeof(double)      * numseeds ) ;
//    kseedsy    = (double*)mxMalloc( sizeof(double)      * numseeds ) ;
//    kseedsl    = (double*)mxMalloc( sizeof(double)      * numseeds ) ;
//    kseedsa    = (double*)mxMalloc( sizeof(double)      * numseeds ) ;
//    kseedsb    = (double*)mxMalloc( sizeof(double)      * numseeds ) ;
//    for(k = 0; k < numseeds; k++)
//    {
//        kseedsx[k] = seedIndices[k]%width;
//        kseedsy[k] = seedIndices[k]/width;
//        kseedsl[k] = lvec[seedIndices[k]];
//        kseedsa[k] = avec[seedIndices[k]];
//        kseedsb[k] = bvec[seedIndices[k]];
//    }
    //---------------------------
    // Compute superpixels
    //---------------------------
    //GetDominantColors(img, lvec, avec, bvec, width, height, klabels, &numklabels);//, avgcolors, closestcolors, clustersizes);
    ClusterWithKnownDistance(lvec, avec, bvec, width, height, klabels, &numklabels, distanceThreshold);
    
    ConnectSpatially(klabels,width,height,clabels,&numclabels, minsz);
    
    
    //---------------------------
    // Assign output labels
    //---------------------------
    plhs[0] = mxCreateNumericMatrix(height,width,mxINT32_CLASS,mxREAL);
    plhs[2] = mxCreateNumericMatrix(height,width,mxINT32_CLASS,mxREAL);
    outklabels = (int*)mxGetData(plhs[0]);
    outclabels = (int*)mxGetData(plhs[2]);
    for(x = 0, ii = 0; x < width; x++)//copying data from row-major C matrix to column-major MATLAB matrix (i.e. perform transpose)
    {
        for(y = 0; y < height; y++)
        {
            i = y*width+x;
            outklabels[ii] = klabels[i];
            outclabels[ii] = clabels[i];
            ii++;
        }
    }
    
    
    //---------------------------
    // Assign number of labels/seeds
    //---------------------------
    plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    numoutklabels = (int*)mxGetData(plhs[1]);//gives a void*, cast it to int*
    *numoutklabels = numklabels;
    plhs[3] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    numoutclabels = (int*)mxGetData(plhs[3]);//gives a void*, cast it to int*
    *numoutclabels = numclabels;
    //---------------------------
    // Deallocate memory
    //---------------------------
    mxFree(rin);
    mxFree(gin);
    mxFree(bin);
    mxFree(lvec);
    mxFree(avec);
    mxFree(bvec);
    mxFree(img);
    mxFree(klabels);
    mxFree(clabels);
//    mxFree(seedIndices);
//    mxFree(kseedsx);
//    mxFree(kseedsy);
//    mxFree(kseedsl);
//    mxFree(kseedsa);
//    mxFree(kseedsb);
    
}
