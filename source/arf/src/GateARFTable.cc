/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "G4SystemOfUnits.hh"

#include "GateARFTable.hh"
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <utility>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH2D.h"
#include "TH1D.h"


//#include "Math/SpecFunc.h"
#include "TMath.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateARFTable::GateARFTable(G4String aName):m_name(aName)
{
m_Index = 0;         // the index of the AERF Table because many are defined for one simulation
m_theTable = 0; // contains the probability for each cos(theta)_k, tan(phi)_k' as a single linearized index
m_theDRFTable = 0;

//m_NbOfcosTheta = 2048 ; // the number of discretized values of cos(theta)

//m_NbOfcosTheta = 2142 ; // the number of discretized values of cos(theta)
//m_rejected = 0;


m_NbOfcosTheta = 2048 ;
m_drfdimx = 1280;
m_drfdimy = 1280;
m_drfbinsize = 0.0442*cm;
m_NbOftanPhi   = 512 ;    // the number of discretized values of tan(phi)
m_TotalNb = m_NbOfcosTheta  * m_NbOftanPhi;

m_counter = 0;
//m_TotPhotons = 0;


iphicounts = 0;

//dStep1 =  0.010 / 1024.0;

//dStep1 = 1./2047.;
//dStep2 = 1./2047.;
//dStep3 = 1./2047.;
//dStep4 = 1./2047.;

dStep1 = 0.010 / 1024.0;
dStep2 = 0.040 / 512.0;
dStep3 = 0.20 / 256.0;
dStep4 = 0.55 / 256.0;


dTanPhiStep1 =  1.0 / 256.0;

dBase1 = 1. + dStep1 / 2.0;
dBase2 = 0.990005;      /* 1024 */
dBase3 = 0.950005;      /* 1536 */
dBase4 = 0.750005;	/* 1792 */  

/*----------------------------------------------------------------------
        variables for energy resolution
        If dErgReso>0, used for energy resolution ~ square_root (photon enerngy)
                static double dEWindowWidth, dErgReso = 10.0, dResoAtErg = 140.5;
        if dErgReso<0, energy resolution linearly depends on photon energy
                for max E > 240 keV: Siemens E.CAM, dConstant=0.971162, dLinear=0.0555509
                for max E < 240 keV: Siemens E.CAM, dConstant=1.9335, dLinear=0.0383678
                OLD(for Siemens E.CAM, dConstant=-0.226416, dLinear=0.0592695)
----------------------------------------------------------------------*/

dErgReso=0.095; // it is a percentage


//dErgReso= 0. ;

dResoAtErg=140.5 * keV;
dConstantTerm=1.9335 * keV;
dLinearTerm=0.0383678;

SetNoPrimary();

m_Elow = 0.0;      // the left end energy of the energy window
m_Ehigh = 0.0;

m_ElowOut = 0.0; // window energy specified by the use
m_EhighOut = 0.0;

m_theta = 0;   
cosTheta = 0;
tanPhi = 0;
Phi = 0;

iAvgPixNum= 5;
fDist_src2img = 36.05 * cm ; // distance from source to the detector projection plane which is the middle of the detector's depth
 }

GateARFTable::~GateARFTable()
{

if ( m_theTable != 0 ) {delete [] m_theTable;}
if ( cosTheta   != 0 ) {delete [] cosTheta;}
if ( tanPhi     != 0 ) {delete [] tanPhi;}
if ( Phi        != 0 ) {delete [] Phi;}

}

void GateARFTable::Initialize(G4double Elow, G4double Ehigh)
{
m_ElowOut = Elow; // window energy specified by the user
m_EhighOut = Ehigh;



 if ( cosTheta !=0 ) { delete [] cosTheta;}
 cosTheta = new G4double[ 2048];
 cosTheta_i = new G4double[ 2048];
m_theta = new G4double[ 2048 ];

G4cout.precision(10);

                 G4double tmp = 1.0 + dStep1;

// tmp = 1.;

 for (G4int i = 0; i < 2048; i++) 
{


 // i is cos(Theta), 1.0 --> 0.20 
                       if (i<1025) { tmp -= dStep1;
                                cosTheta[i] = tmp;
                                m_theta[i] = acos(tmp);
                        } 
	 	        if ( (i > 1024)  && (i < 1537) ) { tmp -= dStep2;
                                cosTheta[i] = tmp;
                                m_theta[i] = acos(tmp);
                        }
			if (  (i > 1536 )  &&  (i < 1793) ) { tmp -= dStep3;
                                cosTheta[i] = tmp;
                                m_theta[i] = acos(tmp);
                        }
                       if (   i > 1792 )  { tmp -= dStep4;
                                cosTheta[i] = tmp;
                                m_theta[i] = acos(tmp);
                        }


//cosTheta[i] = tmp; tmp -= 0.8 / 2047.;

cosTheta_i[ 2047 - i] = cosTheta[i];
//G4cout << i << "  costheta " << cosTheta[i] <<"   theta " << m_theta[i]*180./M_PI<<G4endl;

}

 //for (G4int i = 0; i < 2048; i++) G4cout << i << "  costheta " << cosTheta_i[i] <<G4endl;


 if ( tanPhi !=0 ) { delete [] tanPhi;}
 tanPhi = new G4double[512 ];
tanPhi_i = new G4double[ 256 ];
 if ( Phi !=0 ) { delete [] Phi;}
 Phi = new G4double[ 512 ];

 tmp = - dTanPhiStep1;
 G4double tmpp = 1.;
        for (G4int i = 0; i < 512; i++)
 { /* i is the value of Phi, 0 --> 180 */

               
                if (i<256)      tmp += dTanPhiStep1;
                if (i == 256)  { tmp = 1.0; }
                if ( i > 256 )  { tmp -= dTanPhiStep1;  tmpp += dTanPhiStep1;}

   
    tanPhi[i] = tmp;

    
    if ( i < 256 ) tanPhi_i[i] = tanPhi[i];
    if ( i < 257 ) Phi[i] = atan( tanPhi[i] ); else Phi[i] = atan( tmpp );

//G4cout << i << "  tanphi " << tanPhi[i] <<G4endl;

}

      // for (G4int i = 0; i < 512; i++)G4cout << i << "  tanphi " << tanPhi[i] <<G4endl;



   if ( m_theTable != 0 ){ delete [] m_theTable;}

   m_theTable = new G4double[m_TotalNb];

   m_theDRFTable = new G4double[m_drfdimx * m_drfdimy];

   m_lowX = - .5 * G4double(m_drfdimx) * m_drfbinsize;
   m_lowY = - .5 * G4double(m_drfdimy) * m_drfbinsize;

  for ( G4int i = 0 ; i < m_TotalNb ; i++ ) { m_theTable[i] = 0.;}
  for ( G4int i = 0;i < m_drfdimx * m_drfdimy ; i++ ) m_theDRFTable[i] = 0.;

  G4cout << " Initialized ARF Table " << GetName() << G4endl;


std::stringstream s1,s2;
s1 << int(m_Elow/keV);
s2 << int(m_Ehigh/keV);


}

G4double GateARFTable::RetrieveProbability(G4double x , G4double y )
{ G4int itheta, iphi;
 if ( GetIndexes( x , y , itheta, iphi ) == 1 )  return m_theTable[ itheta + iphi * m_NbOfcosTheta]; 
 return 0.;
}

void GateARFTable::SetEnergyReso(G4double aE)
{ dErgReso = aE;}

void GateARFTable::SetERef(G4double aE)
{ dResoAtErg = aE;}


G4int GateARFTable:: GetIndexes( G4double x, G4double y , G4int& itheta, G4int& iphi )
{

  G4double costheta = sqrt( 1. - x*x - y*y );

  if (costheta < 0.2) return 0;

  if( costheta - 0.99 > 0. )   itheta = G4int( (1.0 - costheta ) * 1024.0 / 0.010 );


  if ( (costheta - 0.95 > 0. ) && (costheta - 0.99 <= 0. ) )   itheta =   G4int( (0.99 - costheta ) * 512.0 / 0.040 ) + 1024;


  if ( (costheta - 0.75 >0.)  && (costheta - 0.95 <= 0. ))  itheta =  G4int( ( 0.95 - costheta ) * 256.0 / 0.20 )  + 1536;

 if ( costheta - 0.75 <= 0.)     itheta =  G4int ( (0.75 - costheta ) * 256.0 / 0.55 )   + 1792;

if ( itheta > 0 && (cosTheta[itheta] - costheta  <  0. )  ) itheta--;
if ( itheta < 2047 &&  ( cosTheta[itheta+1] - costheta > 0. ) ) itheta++;


// itheta = G4int( (1.0 - costheta ) * 2047./0.8  );

//G4cout << cosTheta[itheta]<<"   "<<costheta<< cosTheta[itheta + 1 ]<<G4endl;

if ( itheta == 0 ) { iphi = 0; return 1; }

if ( fabs(x) <= 1.e-8 ) {  iphi = 511;   return 1;}

G4double tanphi = y/x; 

if (tanphi < 0.0) tanphi *= -1.0;

  if(tanphi - 1.00 < 0. ){ iphi  = G4int(  tanphi /dTanPhiStep1 + 0.5 );
                           if ( tanphi - tanPhi[iphi] < 0. ) iphi--;

         //G4cout << " tanphi["<<iphi<<"] =  "<<tanPhi[iphi]<< "   "<< tanphi<<"   tanphi["<<iphi+1<<"] =  "<<tanPhi[iphi+1]<< G4endl;


  }else {
          tanphi = fabs(x/y);G4int ii = G4int(  tanphi /dTanPhiStep1 + 0.5 );
	  if ( tanphi - tanPhi[ii] < 0. ) ii--;
          iphi = 511 - ii;
     // G4cout << " tanphi > 1 "<<G4endl;
     // if ( iphi == 256 ) G4cout << " tanphi["<<iphi<<"] =  "<<tanPhi[iphi]<< "   "<< fabs(y/x)<<"   tanphi["<<iphi-1<<"] =  "<<tanPhi[iphi-1]<<G4endl;

        // G4cout << " tanphi["<<ii<<"] =  "<<tanPhi[ii]<< "   "<< tanphi<<"   tanphi["<<ii+1<<"] =  "<<tanPhi[ii+1]<< G4endl;

         }

 
return 1;

}

G4double GateARFTable::computeARFfromDRF( G4double xi, G4double yj, G4double costheta )
{
	G4int index0;
	G4int i, j, k, l, m, n;
	G4int iStarti, iStartj, iStopi, iStopj;
	G4double dSum=0.0;
	G4double dSize, dFactor;
	G4double dSmallPixX, dSmallPixY;
	G4double dLengthSqr, dRefRadius, dRefRadiusSqr;
	G4int iNumSmallPixSummed = 0;
	
	dRefRadius = ((G4double) iAvgPixNum + 0.5 ) * m_drfbinsize;
	dRefRadiusSqr = dRefRadius * dRefRadius;

	G4int iCenteri = (G4int) (m_drfdimx/2);
	G4int iCenterj = (G4int) (m_drfdimy/2);	
	i = iCenteri + G4int ( xi / m_drfbinsize + 0.5);  /* for 1023x1023. Center is (511,511) */
	j = iCenterj + G4int ( yj / m_drfbinsize + 0.5);

	/* get totally (2iAvgPixNum+1 x 2iAvgPixNum+1) matrix */ 
	iStarti = i -  iAvgPixNum;
	iStopi = i + iAvgPixNum;

	iStartj = j - iAvgPixNum;
	iStopj = j + iAvgPixNum;
/*
printf("\nnow in fCalcProb, iStarti=%d, iStartj=%d, iStopi=%d, iStopj=%d, Cos=%2.10g, iAvgPixNum=%d\n", iStarti, iStartj, iStopi, iStopj,dCosTheta, iAvgPixNum);
printf("dRefRadius=%5.10g\n", dRefRadius);
*/
	
	for (n = iStartj; n<iStopj+1; n++)
	{
		for (m = iStarti; m<iStopi+1; m++)
		{
			index0 = m + n * m_drfdimx;
			
			/* loops k and l are used to check the 10x10 small elements in each big element (m, n). All elements with the distance
			to the photon point (dTmpi, dTmpj) smaller than the user specified radius are used to do average */ 
			for (k=0; k<10; k++)
			{
				dSmallPixX = ( G4double (m-iCenteri) + (G4double( k ) - 4.5 ) / 10.0) * m_drfbinsize;	

				for (l=0; l<10; l++)
				{
					/* dSmallPixX and dSmallPixY are the absolute values for that the small element in unit of cm */
					dSmallPixY = ( G4double (n-iCenterj) + ( G4double( l ) - 4.5 ) / 10.0) * m_drfbinsize;	
					
					/* distance in unit of cm */
					dLengthSqr = (dSmallPixX- xi)*(dSmallPixX- xi) + (dSmallPixY- yj)*(dSmallPixY- yj);
					
					if(dLengthSqr < dRefRadiusSqr) {
						dSum = dSum + m_theDRFTable[index0];
						iNumSmallPixSummed++;
					}
				}  /* end of loop l */
//	printf("dSmallPixX=%5.10g, dSmallPixY=%5.10g, dTmpi=%5.10g, dTmpj=%5.10g\n", dSmallPixX, dSmallPixY, dTmpi, dTmpj);
			}  /* end of loop k */
		}  /* end of loop m */
	}  /* end of loop n */

//printf("dSum = %5.10g, iNumSmallPixSummed=%d\n", dSum, iNumSmallPixSummed);

	dSum /= 100.0;
	dSize = m_drfbinsize * m_drfbinsize * G4double( iNumSmallPixSummed )/ 100.0;


	
	dFactor = fDist_src2img * fDist_src2img / (dSize * costheta * costheta *  costheta);
	dFactor = dFactor * 4.0 * M_PI / m_TotSimuPhotons; 
	
	
	dSum *= dFactor;
	
//printf(" fSum1=%2.10g, fSum2=%2.10g, fSum=%2.10g, dFactor=%5.10g, fsize=%5.10g\n", fSum1, fSum2,fSum, dFactor, dSize);
	return dSum;
	
}
 void  GateARFTable::convertDRF2ARF()
{
G4double dHalfTblRangeInCM_X,dHalfTblRangeInCM_Y;
G4double cosphi, sinphi, i , j;
G4int index1,index2,index3,index4;

// prepare the DRF table before processing
	for(G4int j=0; j< m_drfdimy; j++) {
		for(G4int i=0; i< m_drfdimx; i++) {
			index1= i+j*m_drfdimx;
			index2= (m_drfdimx - i -1 ) + j * m_drfdimx;
			index3= (m_drfdimx - i -1 ) + (m_drfdimy - j -1) * m_drfdimx;
			index4= i + (m_drfdimy - j -1 ) * m_drfdimx;
			m_theDRFTable[index1] = m_theDRFTable[index1] + m_theDRFTable[index2] + m_theDRFTable[index3] +m_theDRFTable[index4];
			m_theDRFTable[index1] /= 4.0;
		}
	}

	dHalfTblRangeInCM_X = (m_drfdimx/2.0-iAvgPixNum-2.0)*m_drfbinsize;
	dHalfTblRangeInCM_Y = (m_drfdimy/2.0-iAvgPixNum-2.0)*m_drfbinsize;

	for (G4int iphi = 0; iphi < 512; iphi++)
 {	
   if ( iphi ==0 ) {
			sinphi = 0.0;
			cosphi = 1.0;
		} else if (iphi <256 ) {      /* in fact, this is the real tan(PHI) */
			
			cosphi = 1.0 / sqrt(1.0 + tanPhi[iphi] * tanPhi[iphi] );
			sinphi = cosphi * tanPhi[iphi];
			
		} else if (iphi == 256) {  /* in fact, from 256-511, the actualy PHI value is for ctg, not for tan */
			sinphi = sqrt(2.0)/2.0;
			cosphi = sinphi;
		} else {
			//dTanPhi -= dTanPhiStep1;  /* the value of dTanPhi is actually the value of ctan of the same angle */
			sinphi = 1.0 / sqrt(1.0 + tanPhi[512 - iphi]  * tanPhi[512 - iphi] );
			cosphi = sinphi * tanPhi[512 - iphi];

		}
	for (G4int itheta = 0; itheta < 2048; itheta++) { /* x is cos(Theta), 1.0 --> 0.20 */
			
			G4int index = itheta + iphi * 2048; 

			if (itheta == 0 ) {
				i = 0.0;
				j = 0.0;
			} else {
				G4double dR = fDist_src2img * sqrt(1.0 / ( cosTheta[itheta] * cosTheta[itheta] ) - 1.0);
				i = dR * cosphi;
				j = dR * sinphi;
			}
			
			if (( i >= dHalfTblRangeInCM_X) || ( j >= dHalfTblRangeInCM_Y)) {
		   	m_theTable[index] = 0.0;
		   } else {
				m_theTable[index] = computeARFfromDRF( i, j, cosTheta[itheta] );
			}
		}
	}

G4String m_fn = GetName()+"_ARFfromDRFTable.bin";
size_t theBufferSize = m_TotalNb*sizeof(G4double);
std::ofstream destbin ( m_fn.c_str(), std::ios::out | std::ios::binary );
destbin.write((const char*)( m_theTable ), theBufferSize );
destbin.close();

G4cout << " writing the ARF table to a text file " << G4endl;
std::ofstream dest ( "arftable.txt");
for (G4int i = 0;i <m_TotalNb; i++ )
{ G4int iphi = i/GetNbofTheta();
  G4int itheta = i - iphi * GetNbofTheta();
dest <<iphi<<" "<<itheta<<"  "<<m_theTable[i]<<G4endl;
if ( itheta == 2047 ) dest <<"  "<<G4endl;
}

}


void  GateARFTable::FillDRFTable( G4double dMeanE, G4double X , G4double Y )
{


if ( X - m_lowX < 0. || X + m_lowX > 0. ) return;
if ( Y - m_lowY < 0. || Y + m_lowY > 0. ) return;

 G4int ix = G4int ( ( X - m_lowX ) / m_drfbinsize );
 G4int iy = G4int ( ( Y - m_lowY ) / m_drfbinsize );

if ( ix > m_drfdimx - 1 || ( ix < 0 ) ) return;
if ( iy > m_drfdimy - 1 || ( iy < 0 ) ) return;

//G4cout << "X " <<X<<"   X pixelllized "<< G4double ( ix * m_drfbinsize ) + m_lowX<<"         Y "<<Y<<" Y pixelllized "<< G4double ( iy * m_drfbinsize ) + m_lowX<<G4endl;

G4int index = ix + m_drfdimx * iy;


 m_counter++;

/*------------------------------------------------------------------------------
   erf(x)=1/sqrt(PI) * integral(-x,x) of exp(-x^2)
         =2/sqrt(PI) * integral(0,x) of exp(-x^2)
   assume Gaussian exp(-x^2), where x^2 = (u/sig)^2
                sig = fwhm / (2.0*sqrt(log(2.0)))
        EnergyResolution (ErgReso) is considered 0.0 if -10^(-6)<ErgReso<10^(-6)
-------------------------------------------------------------------------------*/
  G4double fwhm, sig;
  G4double result = 0.;

  if( dErgReso>0.000001 )
  {
    /* energy resolution (in %) ~ sqrt(E), relative to dErgReso@dResoAtErg kev*/

    fwhm = dErgReso* sqrt( dResoAtErg * dMeanE );
    sig = fwhm / (2.0*sqrt(log(2.0)));

   result = ( TMath::Erf( (m_EhighOut-dMeanE)/sig ) - TMath::Erf( (m_ElowOut-dMeanE)/sig) ) / 2.0;


  }
   else if( dErgReso<-0.000001 ){
    /* Gaussian width of photopeaks linearly depends on photon energy */
    sig = dConstantTerm + dLinearTerm * dMeanE;
   result = ( TMath::Erf( (m_EhighOut-dMeanE)/sig ) - TMath::Erf( (m_ElowOut-dMeanE)/sig) ) / 2.0;
  }
   else if (dMeanE>= m_ElowOut && dMeanE< m_EhighOut)
  {
    /*perfect energy resolution*/
                  result=1.0;

//  G4cout  << m_ElowOut<<"   "<<dMeanE<<"  " <<m_EhighOut <<"  "<<result<<G4endl;

  }
m_theDRFTable[index]+=fabs(result);
}




void GateARFTable::Describe()
{
G4cout << "===== Description of the ARF Table named " << GetName() << " ======" << G4endl;
G4cout << "      Index                  " << GetIndex() << G4endl;
G4cout << "      Number of Theta Angles " <<  GetNbofTheta()<<G4endl;
G4cout << "      Number of Phi Angles   " <<  GetNbofPhi()<<G4endl;
G4cout << "      Total number of values " <<  GetTotalNb() << G4endl;
G4cout << "      Energy Window                 (keV)[" <<  GetElow()/keV << " , " << GetEhigh()/keV << "]"<<G4endl;
G4cout << "      User Specified Energy Window  (keV)[" <<  m_ElowOut/keV << " , " << m_EhighOut/keV << "]"<<G4endl;
G4cout << "      Is Primary             " << GetPrimary() << G4endl;
G4cout << "      Energy Resolution " << dErgReso <<G4endl;
G4cout << "      Energy Of Reference " <<dResoAtErg<<G4endl;
G4cout << "      Energy Deposition Threshold " << m_ElowOut/keV<<" keV"<<G4endl;
G4cout << "      Energy Deposition Uphold " << m_EhighOut/keV<<" keV"<<G4endl;

//G4cout << "      Total Number Of Photons for the Energy Window " << m_TotPhotons << G4endl;
G4cout << "      Total Number of Binned Photons (not necessarily detected in the energy window chosen) " <<  m_counter <<G4endl;
//G4cout << "      number of REJECTED photons " << m_rejected <<G4endl;


}


void GateARFTable::GetARFAsBinaryBuffer(G4double*& theBuffer)
{

  theBuffer[0] = G4double( GetElow() );
  theBuffer[1] = G4double( GetEhigh() );
  theBuffer[2] = G4double( GetEnergyReso() );
  theBuffer[3] = G4double( GetERef() );
  theBuffer[4] = G4double( GetEWlow() );
  theBuffer[5] = G4double( GetEWhigh() );
  for (int i = 0;i< GetTotalNb();i++ )
  theBuffer[i+6] = m_theTable[i];

}


void GateARFTable::FillTableFromBuffer(G4double*& theBuffer)
{
 m_Elow = theBuffer[0];
 m_Ehigh = theBuffer[1];
 dErgReso = theBuffer[2];
 dResoAtErg = theBuffer[3];
 m_ElowOut = theBuffer[4];
 m_EhighOut = theBuffer[5];

//  SetElow(theBuffer[1]);
//  SetEhigh(theBuffer[2]);
//  SetEnergyReso(theBuffer[3]);
//  SetERef(theBuffer[4]);



  for (int i = 0;i< GetTotalNb();i++ )
  {m_theTable[i] = theBuffer[i+6];}

}

 #endif
