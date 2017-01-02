/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See GATE/LICENSE.txt for further details
 ------*/

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
#include "TMath.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateARFTable::GateARFTable(const G4String & aName) :
    mArfTableName(aName)
  {
  mArfTableIndex = 0; /* the index of the AERF Table because many are defined for one simulation */
  mArfTableVector = 0; /* contains the probability for each cos(theta)_k, tan(phi)_k' as a single linearized index */
  mDrfTableVector = 0;
  mNbOfCosTheta = 2048;
  m_drfdimx = 1280;
  mDrfDimY = 1280;
  mDrfBinSize = 0.0442 * cm;
  mNbOfTanPhi = 512; /* the number of discretized values of tan(phi) */
  mTotalNbOfThetaPhi = mNbOfCosTheta * mNbOfTanPhi;
  mBinnedPhotonCounter = 0;
  mPhiCounts = 0;
  mStep1 = 0.010 / 1024.0;
  mStep2 = 0.040 / 512.0;
  mStep3 = 0.20 / 256.0;
  mStep4 = 0.55 / 256.0;
  mTanPhiStep = 1.0 / 256.0;
  mBase1 = 1. + mStep1 / 2.0;
  mBase2 = 0.990005; /* 1024 */
  mBase3 = 0.950005; /* 1536 */
  mBase4 = 0.750005; /* 1792 */
  /*
   variables for energy resolution
   If dErgReso>0, used for energy resolution ~ square_root (photon enerngy)
   static double dEWindowWidth, dErgReso = 10.0, dResoAtErg = 140.5;
   if dErgReso<0, energy resolution linearly depends on photon energy
   for max E > 240 keV: Siemens E.CAM, dConstant=0.971162, dLinear=0.0555509
   for max E < 240 keV: Siemens E.CAM, dConstant=1.9335, dLinear=0.0383678
   OLD(for Siemens E.CAM, dConstant=-0.226416, dLinear=0.0592695)
   */
  mEnergyResolution = 0.095; /* it is a percentage */
  mEnergyReference = 140.5 * keV;
  mConstantTerm = 1.9335 * keV;
  mLinearTerm = 0.0383678;
  SetNoPrimary();
  mELow = 0.0; /* the left end energy of the energy window */
  mEHigh = 0.0;
  mEnergyLowOut = 0.0; /* window energy specified by the use */
  mEnergyHighOut = 0.0;
  mTheta = 0;
  mCosTheta = 0;
  mTanPhi = 0;
  mPhi = 0;
  mAvgPixNum = 5;
  mfDistSrc2Img = 36.05 * cm; /* distance from source to the detector projection plane which is the middle of the detector's depth */
  }

GateARFTable::~GateARFTable()
  {
  if (mArfTableVector != 0)
    {
    delete[] mArfTableVector;
    }
  if (mCosTheta != 0)
    {
    delete[] mCosTheta;
    }
  if (mTanPhi != 0)
    {
    delete[] mTanPhi;
    }
  if (mPhi != 0)
    {
    delete[] mPhi;
    }
  }

void GateARFTable::Initialize(const G4double & energyLow, const G4double & energyHigh)
  {
  mEnergyLowOut = energyLow; /* window energy specified by the user */
  mEnergyHighOut = energyHigh;
  if (mCosTheta != 0)
    {
    delete[] mCosTheta;
    }
  mCosTheta = new G4double[2048];
  mCosThetaI = new G4double[2048];
  mTheta = new G4double[2048];
  G4cout.precision(10);
  G4double tmp = 1.0 + mStep1;
  for (G4int cosTheta = 0; cosTheta < 2048; cosTheta++)
    {
    /* cosTheta 1.0 --> 0.20 */
    if (cosTheta < 1025)
      {
      tmp -= mStep1;
      mCosTheta[cosTheta] = tmp;
      mTheta[cosTheta] = acos(tmp);
      }
    if ((cosTheta > 1024) && (cosTheta < 1537))
      {
      tmp -= mStep2;
      mCosTheta[cosTheta] = tmp;
      mTheta[cosTheta] = acos(tmp);
      }
    if ((cosTheta > 1536) && (cosTheta < 1793))
      {
      tmp -= mStep3;
      mCosTheta[cosTheta] = tmp;
      mTheta[cosTheta] = acos(tmp);
      }
    if (cosTheta > 1792)
      {
      tmp -= mStep4;
      mCosTheta[cosTheta] = tmp;
      mTheta[cosTheta] = acos(tmp);
      }
    mCosThetaI[2047 - cosTheta] = mCosTheta[cosTheta];
    }
  if (mTanPhi != 0)
    {
    delete[] mTanPhi;
    }
  mTanPhi = new G4double[512];
  mTanPhiI = new G4double[256];
  if (mPhi != 0)
    {
    delete[] mPhi;
    }
  mPhi = new G4double[512];
  tmp = -mTanPhiStep;
  G4double tmpp = 1.;
  for (G4int phi = 0; phi < 512; phi++)
    { /*  Phi, 0 --> 180 */
    if (phi < 256)
      tmp += mTanPhiStep;
    if (phi == 256)
      {
      tmp = 1.0;
      }
    if (phi > 256)
      {
      tmp -= mTanPhiStep;
      tmpp += mTanPhiStep;
      }
    mTanPhi[phi] = tmp;
    if (phi < 256)
      {
      mTanPhiI[phi] = mTanPhi[phi];
      }
    if (phi < 257)
      {
      mPhi[phi] = atan(mTanPhi[phi]);
      }
    else
      {
      mPhi[phi] = atan(tmpp);
      }
    }
  if (mArfTableVector != 0)
    {
    delete[] mArfTableVector;
    }
  mArfTableVector = new G4double[mTotalNbOfThetaPhi];
  mDrfTableVector = new G4double[m_drfdimx * mDrfDimY];
  mLowX = -.5 * G4double(m_drfdimx) * mDrfBinSize;
  mLowY = -.5 * G4double(mDrfDimY) * mDrfBinSize;
  for (G4int i = 0; i < mTotalNbOfThetaPhi; i++)
    {
    mArfTableVector[i] = 0.;
    }
  for (G4int i = 0; i < m_drfdimx * mDrfDimY; i++)
    {
    mDrfTableVector[i] = 0.;
    }
  G4cout << " Initialized ARF Table " << GetName() << Gateendl;
  }

G4double GateARFTable::RetrieveProbability(const G4double & x, const G4double & y)
  {
  G4int theta = 0;
  G4int phi;
  if (GetIndexes(x, y, theta, phi) == 1)
    {
    return mArfTableVector[theta + phi * mNbOfCosTheta];
    }
  return 0.;
  }

void GateARFTable::SetEnergyReso(const G4double & aE)
  {
  mEnergyResolution = aE;
  }

void GateARFTable::SetERef(const G4double & aE)
  {
  mEnergyReference = aE;
  }

G4int GateARFTable::GetIndexes(const G4double & x, const G4double & y, G4int& theta, G4int& phi)
  {
  G4double costheta = sqrt(1. - x * x - y * y);
  if (costheta < 0.2)
    {
    return 0;
    }
  if (costheta - 0.99 > 0.)
    {
    theta = G4int((1.0 - costheta) * 1024.0 / 0.010);
    }

  if ((costheta - 0.95 > 0.) && (costheta - 0.99 <= 0.))
    {
    theta = G4int((0.99 - costheta) * 512.0 / 0.040) + 1024;
    }

  if ((costheta - 0.75 > 0.) && (costheta - 0.95 <= 0.))
    {
    theta = G4int((0.95 - costheta) * 256.0 / 0.20) + 1536;
    }

  if (costheta - 0.75 <= 0.)
    {
    theta = G4int((0.75 - costheta) * 256.0 / 0.55) + 1792;
    }

  if (theta > 0 && (mCosTheta[theta] - costheta < 0.))
    {
    theta--;
    }
  if (theta < 2047 && (mCosTheta[theta + 1] - costheta > 0.))
    {
    theta++;
    }
  if (theta == 0)
    {
    phi = 0;
    return 1;
    }
  if (fabs(x) <= 1.e-8)
    {
    phi = 511;
    return 1;
    }
  G4double tanPhi = y / x;
  if (tanPhi < 0.0)
    {
    tanPhi *= -1.0;
    }

  if (tanPhi - 1.00 < 0.)
    {
    phi = G4int(tanPhi / mTanPhiStep + 0.5);
    if (tanPhi - mTanPhi[phi] < 0.)
      {
      phi--;
      }
    }
  else
    {
    tanPhi = fabs(x / y);
    G4int tanPhiIndex = G4int(tanPhi / mTanPhiStep + 0.5);
    if (tanPhi - mTanPhi[tanPhiIndex] < 0.)
      {
      tanPhiIndex--;
      }
    phi = 511 - tanPhiIndex;
    }
  return 1;
  }

G4double GateARFTable::computeARFfromDRF(const G4double & xI,
                                         const G4double & yJ,
                                         const G4double & cosTheta)
  {
  G4int index0;
  G4int i;
  G4int j;
  G4double sum = 0.0;

  G4double smallPixX;
  G4double smallPixY;
  G4double lengthSqr;
  G4double refRadius = ((G4double) mAvgPixNum + 0.5) * mDrfBinSize;
  G4double refRadiusSqr = refRadius * refRadius;
  G4int numSmallPixSummed = 0;
  G4int centerI = (G4int) (m_drfdimx / 2);
  G4int centerJ = (G4int) (mDrfDimY / 2);
  i = centerI + G4int(xI / mDrfBinSize + 0.5); /* for 1023x1023. Center is (511,511) */
  j = centerJ + G4int(yJ / mDrfBinSize + 0.5);
  /* get totally (2iAvgPixNum+1 x 2iAvgPixNum+1) matrix */
  for (G4int n = j - mAvgPixNum; n < j + mAvgPixNum + 1; n++)
    {
    for (G4int m = i - mAvgPixNum; m < i + mAvgPixNum + 1; m++)
      {
      index0 = m + n * m_drfdimx;
      /* loops k and l are used to check the 10x10 small elements in each big element (m, n). All elements with the distance
       to the photon point (dTmpi, dTmpj) smaller than the user specified radius are used to do average */
      for (G4int k = 0; k < 10; k++)
        {
        smallPixX = (G4double(m - centerI) + (G4double(k) - 4.5) / 10.0) * mDrfBinSize;
        for (G4int l = 0; l < 10; l++)
          {
          /* dSmallPixX and dSmallPixY are the absolute values for that the small element in unit of cm */
          smallPixY = (G4double(n - centerJ) + (G4double(l) - 4.5) / 10.0) * mDrfBinSize;
          /* distance in unit of cm */
          lengthSqr = (smallPixX - xI) * (smallPixX - xI) + (smallPixY - yJ) * (smallPixY - yJ);
          if (lengthSqr < refRadiusSqr)
            {
            sum = sum + mDrfTableVector[index0];
            numSmallPixSummed++;
            }
          } /* end of loop l */
        } /* end of loop k */
      } /* end of loop m */
    } /* end of loop n */
  sum /= 100.0;
  G4double size = mDrfBinSize * mDrfBinSize * G4double(numSmallPixSummed) / 100.0;
  G4double factor = mfDistSrc2Img * mfDistSrc2Img / (size * cosTheta * cosTheta * cosTheta);
  factor *= (4.0 * pi / mTotalNumberOfPhotons);
  sum *= factor;
  return sum;

  }
void GateARFTable::convertDRF2ARF()
  {

  G4int index1;
  G4int index2;
  G4int index3;
  G4int index4;
  /* prepare the DRF table before processing */
  for (G4int j = 0; j < mDrfDimY; j++)
    {
    for (G4int i = 0; i < m_drfdimx; i++)
      {
      index1 = i + j * m_drfdimx;
      index2 = (m_drfdimx - i - 1) + j * m_drfdimx;
      index3 = (m_drfdimx - i - 1) + (mDrfDimY - j - 1) * m_drfdimx;
      index4 = i + (mDrfDimY - j - 1) * m_drfdimx;
      mDrfTableVector[index1] = mDrfTableVector[index1]
                                + mDrfTableVector[index2]
                                + mDrfTableVector[index3]
                                + mDrfTableVector[index4];
      mDrfTableVector[index1] *= 0.25;
      }
    }
  G4double cosPhi;
  G4double sinPhi;
  G4double halfTblRangeInCmX = (m_drfdimx * 0.5 - mAvgPixNum - 2.0) * mDrfBinSize;
  G4double halfTblRangeInCmY = (mDrfDimY * 0.5 - mAvgPixNum - 2.0) * mDrfBinSize;
  G4double xI;
  G4double yJ;
  G4int index;
  G4double radius;
  for (G4int phiIndex = 0; phiIndex < 512; phiIndex++)
    {
    if (phiIndex == 0)
      {
      sinPhi = 0.0;
      cosPhi = 1.0;
      }
    else if (phiIndex < 256)
      {
      /* in fact, this is the real tan(PHI) */
      cosPhi = 1.0 / sqrt(1.0 + mTanPhi[phiIndex] * mTanPhi[phiIndex]);
      sinPhi = cosPhi * mTanPhi[phiIndex];

      }
    else if (phiIndex == 256)
      {
      /* in fact, from 256-511, the actualy PHI value is for ctg, not for tan */
      sinPhi = sqrt(2.0) * 0.5;
      cosPhi = sinPhi;
      }
    else
      {
      /* the value of dTanPhi is actually the value of ctan of the same angle */
      sinPhi = 1.0 / sqrt(1.0 + mTanPhi[512 - phiIndex] * mTanPhi[512 - phiIndex]);
      cosPhi = sinPhi * mTanPhi[512 - phiIndex];
      }
    for (G4int thetaIndex = 0; thetaIndex < 2048; thetaIndex++)
      {
      /* x is cos(Theta), 1.0 --> 0.20 */
      index = thetaIndex + phiIndex * 2048;
      if (thetaIndex == 0)
        {
        xI = 0.0;
        yJ = 0.0;
        }
      else
        {
        radius = mfDistSrc2Img * sqrt(1.0 / (mCosTheta[thetaIndex] * mCosTheta[thetaIndex]) - 1.0);
        xI = radius * cosPhi;
        yJ = radius * sinPhi;
        }
      if ((xI >= halfTblRangeInCmX) || (yJ >= halfTblRangeInCmY))
        {
        mArfTableVector[index] = 0.0;
        }
      else
        {
        mArfTableVector[index] = computeARFfromDRF(xI, yJ, mCosTheta[thetaIndex]);
        }
      }
    }

  G4String arfDrfTableBinName = GetName() + "_ARFfromDRFTable.bin";
  size_t tableBufferSize = mTotalNbOfThetaPhi * sizeof(G4double);
  std::ofstream outputTableBin(arfDrfTableBinName.c_str(), std::ios::out | std::ios::binary);
  outputTableBin.write((const char*) (mArfTableVector), tableBufferSize);
  outputTableBin.close();
  G4cout << " writing the ARF table to a text file \n";
  std::ofstream outputArtTableTxt("arftable.txt");
  G4int phiIndex;
  G4int thetaIndex;
  for (G4int i = 0; i < mTotalNbOfThetaPhi; i++)
    {
    phiIndex = i / GetNbofTheta();
    thetaIndex = i - phiIndex * GetNbofTheta();
    outputArtTableTxt << phiIndex << " " << thetaIndex << "  " << mArfTableVector[i] << Gateendl;
    if (thetaIndex == 2047)
      {
      outputArtTableTxt << Gateendl;}
    }
  outputArtTableTxt.close();
  }

void GateARFTable::FillDRFTable(const G4double & meanE, const G4double & X, const G4double & Y)
  {
  if (X - mLowX < 0. || X + mLowX > 0.)
    {
    return;
    }
  if (Y - mLowY < 0. || Y + mLowY > 0.)
    {
    return;
    }

  G4int xIndex = G4int((X - mLowX) / mDrfBinSize);
  G4int yIndex = G4int((Y - mLowY) / mDrfBinSize);
  if (xIndex > m_drfdimx - 1 || (xIndex < 0))
    {
    return;
    }
  if (yIndex > mDrfDimY - 1 || (yIndex < 0))
    {
    return;
    }
  G4int index = xIndex + m_drfdimx * yIndex;
  mBinnedPhotonCounter++;
  /*
   erf(x)=1/sqrt(PI) * integral(-x,x) of exp(-x^2)
   =2/sqrt(PI) * integral(0,x) of exp(-x^2)
   assume Gaussian exp(-x^2), where x^2 = (u/sig)^2
   sig = fwhm / (2.0*sqrt(log(2.0)))
   EnergyResolution (ErgReso) is considered 0.0 if -10^(-6)<ErgReso<10^(-6)
   */
  G4double fwhm;
  G4double sigma;
  G4double result = 0.;
  if (mEnergyResolution > 0.000001)
    {
    /* energy resolution (in %) ~ sqrt(E), relative to dErgReso@dResoAtErg kev*/
    fwhm = mEnergyResolution * sqrt(mEnergyReference * meanE);
    sigma = fwhm / (2.0 * sqrt(log(2.0)));
    result = (TMath::Erf((mEnergyHighOut - meanE) / sigma)
              - TMath::Erf((mEnergyLowOut - meanE) / sigma))
             * 0.5;
    }
  else if (mEnergyResolution < -0.000001)
    {
    /* Gaussian width of photopeaks linearly depends on photon energy */
    sigma = mConstantTerm + mLinearTerm * meanE;
    result = (TMath::Erf((mEnergyHighOut - meanE) / sigma)
              - TMath::Erf((mEnergyLowOut - meanE) / sigma))
             * 0.5;
    }
  else if (meanE >= mEnergyLowOut && meanE < mEnergyHighOut)
    {
    /* perfect energy resolution*/
    result = 1.0;
    }
  mDrfTableVector[index] += fabs(result);
  }

void GateARFTable::Describe()
  {
  G4cout << "===== Description of the ARF Table named " << GetName() << " ======\n";
  G4cout << "      Index                  " << GetIndex() << Gateendl;
  G4cout << "      Number of Theta Angles " << GetNbofTheta() << Gateendl;
  G4cout << "      Number of Phi Angles   " << GetNbofPhi() << Gateendl;
  G4cout << "      Total number of values " << GetTotalNb() << Gateendl;
  G4cout << "      Energy Window                 (keV)["
         << GetElow() / keV
         << " , "
         << GetEhigh() / keV
         << "]\n";
  G4cout << "      User Specified Energy Window  (keV)["
         << mEnergyLowOut / keV
         << " , "
         << mEnergyHighOut / keV
         << "]\n";
  G4cout << "      Is Primary             " << GetPrimary() << Gateendl;
  G4cout << "      Energy Resolution " << mEnergyResolution << Gateendl;
  G4cout << "      Energy Of Reference " << mEnergyReference << Gateendl;
  G4cout << "      Energy Deposition Threshold " << mEnergyLowOut / keV << " keV\n";
  G4cout << "      Energy Deposition Uphold " << mEnergyHighOut / keV << " keV\n";

  G4cout << "      Total Number of Binned Photons (not necessarily detected in the energy window chosen) "
         << mBinnedPhotonCounter
         << Gateendl;
  }

void GateARFTable::GetARFAsBinaryBuffer(G4double*& tableBuffer)
  {
  tableBuffer[0] = G4double(GetElow());
  tableBuffer[1] = G4double(GetEhigh());
  tableBuffer[2] = G4double(GetEnergyReso());
  tableBuffer[3] = G4double(GetERef());
  tableBuffer[4] = G4double(GetEWlow());
  tableBuffer[5] = G4double(GetEWhigh());
  for (int i = 0; i < GetTotalNb(); i++)
    {
    tableBuffer[i + 6] = mArfTableVector[i];
    }
  }

void GateARFTable::FillTableFromBuffer(G4double*& tableBuffer)
  {
  mELow = tableBuffer[0];
  mEHigh = tableBuffer[1];
  mEnergyResolution = tableBuffer[2];
  mEnergyReference = tableBuffer[3];
  mEnergyLowOut = tableBuffer[4];
  mEnergyHighOut = tableBuffer[5];
  for (int i = 0; i < GetTotalNb(); i++)
    {
    mArfTableVector[i] = tableBuffer[i + 6];
    }
  }

#endif
