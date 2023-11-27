#ifndef GATEFIXEDFORCEDDETECTIONACTORFUNCTORS_HH
#define GATEFIXEDFORCEDDETECTIONACTORFUNCTORS_HH

/* Geant4 */
#include <G4VEMDataSet.hh>
#include <G4EmCalculator.hh>
#include <G4VDataSetAlgorithm.hh>
#include <G4LivermoreComptonModel.hh>
#include <G4LogLogInterpolation.hh>
#include <G4CompositeEMDataSet.hh>
#include <G4CrossSectionHandler.hh>
#include <G4Poisson.hh>
#include <G4Gamma.hh>
#include <G4PhysicsFreeVector.hh>

/* Gate */
#include "GateEnergyResponseFunctor.hh"

/* ITK */
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <itkTimeProbe.h>
#include <itkForwardFFTImageFilter.h>

/* RTK */
#include <rtkConfiguration.h>

/* XRAYLIB */
#ifdef GATE_USE_XRAYLIB
#include <xraylib.h>
// xraylib defines ZMAX which interfers with another Geant4 dev, undefine.
#  ifdef ZMAX
#    undef ZMAX
#  endif
#endif

#ifndef ITK_MAX_THREADS
using itk::ITK_MAX_THREADS;
#endif

struct newPhoton
  {
  G4ThreeVector direction;
  G4ThreeVector position;
  double weight;
  double energy;
  };

namespace GateFixedForcedDetectionFunctor
  {

    /* Handling of the interpolation weight in primary: store the weights and
     the material indices in vectors and return nada. The integral is computed in the
     ProjectedValueAccumulation since one has to repeat the same ray cast for each
     and every energy of the primary. */
    class InterpolationWeightMultiplication
      {
    public:
      typedef itk::Vector<double, 3> VectorType;

      InterpolationWeightMultiplication()
        {
        }

      ~InterpolationWeightMultiplication()
        {
        }

      bool operator!=(const InterpolationWeightMultiplication &) const
        {
        return false;
        }
      bool operator==(const InterpolationWeightMultiplication & other) const
        {
        return !(*this != other);
        }

      inline double operator()(const rtk::ThreadIdType threadId,
                               const double stepLengthInVoxel,
                               const double weight,
                               const float *p,
                               const int i)
        {
        m_InterpolationWeights[threadId][(int) (p[i])] += stepLengthInVoxel * weight;
        return 0.;
        }

      std::vector<double>* GetInterpolationWeights()
        {
        return m_InterpolationWeights;
        }

    private:
      std::vector<double> m_InterpolationWeights[ITK_MAX_THREADS];
      };

    /* Mother class for accumulation. Purely virtual (FIXME). */
    class VAccumulation
      {
    public:
      static const unsigned int Dimension = 3;
      typedef itk::Vector<double, Dimension> VectorType;
      typedef float InputPixelType;
      typedef itk::Image<InputPixelType, Dimension> InputImageType;
      typedef itk::Image<double, 2> MaterialMuImageType;
      typedef itk::Image<double, 2> MaterialDeltaImageType;

      VAccumulation() :
          m_NumberOfPrimaries(0),
          m_MuToDeltaImageOffset(0),
          m_EnergyResolvedBinSize(0.),
          m_generatePhotons(false),
          m_ARF(false)
        {
        for (itk::ThreadIdType i = 0; i < ITK_MAX_THREADS; i++)
          {
          m_IntegralOverDetector[i] = 0.;
          m_SquaredIntegralOverDetector[i] = 0.;
          }
        }

      bool operator!=(const VAccumulation &) const
        {
        return false;
        }

      bool operator==(const VAccumulation & other) const
        {
        return !(*this != other);
        }
      void SetVolumeSpacing(const VectorType &_arg)
        {
        m_VolumeSpacing = _arg;
        }
      void SetInterpolationWeights(std::vector<double> *_arg)
        {
        m_InterpolationWeights = _arg;
        }
      void SetEnergyWeightList(std::vector<double> *_arg)
        {
        m_EnergyWeightList = _arg;
        }
      void SetMuToDeltaImageOffset(const std::ptrdiff_t o)
        {
        m_MuToDeltaImageOffset = o;
        }

      void Init(unsigned int nthreads)
        {
        for (unsigned int i = 0; i < nthreads; i++)
          {
          m_InterpolationWeights[i].resize(m_MaterialMu->GetLargestPossibleRegion().GetSize()[0]);
          std::fill(m_InterpolationWeights[i].begin(), m_InterpolationWeights[i].end(), 0.);
          }
        }

      /* Solid angle from the source to pixel vector in voxels */
      void SetSolidAngleParameters(const InputImageType::Pointer proj,
                                   const VectorType & u,
                                   const VectorType & v)
        {
        m_DetectorOrientationTimesPixelSurface = proj->GetSpacing()[0]
                                                 * proj->GetSpacing()[1]
                                                 * itk::CrossProduct(u, v);
        }
      double GetSolidAngle(const VectorType & sourceToPixelInVox) const
        {
        VectorType sourceToPixelInMM;
        for (int i = 0; i < 3; i++)
          {
          sourceToPixelInMM[i] = sourceToPixelInVox[i] * m_VolumeSpacing[i];
          }
        return std::abs(sourceToPixelInMM * m_DetectorOrientationTimesPixelSurface
                        / pow(sourceToPixelInMM.GetNorm(), 3.));
        }

      MaterialMuImageType *GetMaterialMu()
        {
        return m_MaterialMu.GetPointer();
        }

      MaterialDeltaImageType *GetMaterialDelta()
        {
        return m_MaterialDelta.GetPointer();
        }

      void CreateMaterialMuMap(G4EmCalculator *emCalculator,
                               const double energySpacing,
                               const double energyMax,
                               GateVImageVolume * gateImageVolume)
        {
        std::vector<double> energyList;
        energyList.push_back(0.);
        while (energyList.back() < energyMax)
          {
          energyList.push_back(energyList.back() + energySpacing);
          }
        CreateMaterialMuMap(emCalculator, energyList, gateImageVolume);
        MaterialMuImageType::PointType origin;
        origin.Fill(0.);
        m_MaterialMu->SetOrigin(origin);
        }

      void CreateMaterialMuMap(G4EmCalculator *emCalculator,
                               const std::vector<double> & energyList,
                               GateVImageVolume * gateImageVolume)
        {
        m_EnergyList = energyList;
        itk::TimeProbe muProbe;
        muProbe.Start();
        /* Get image materials + world */
        std::vector<G4Material*> imageWorldMaterials;
        gateImageVolume->BuildLabelToG4MaterialVector(imageWorldMaterials);
        GateVVolume *volume = gateImageVolume;
        while (volume->GetLogicalVolumeName() != "world_log")
          {
          volume = volume->GetParentVolume();
          }
        imageWorldMaterials.push_back(const_cast<G4Material*>(volume->GetMaterial()));

        /* Get the list of involved processes (Rayleigh, Compton, PhotoElectric) */
        G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle("gamma");
        G4ProcessVector* processList = particle->GetProcessManager()->GetProcessList();
        std::vector<G4String> processNameVector;
        for (unsigned int process = 0; process < processList->size(); process++)
          {
          G4ProcessType type = (*processList)[process]->GetProcessType();
          std::string name = (*processList)[process]->GetProcessName();
          if ((type == fElectromagnetic) && (name != "msc"))
            {
            processNameVector.push_back(name);
            }
          }

        MaterialMuImageType::RegionType region;
        region.SetSize(0, imageWorldMaterials.size());
        region.SetSize(1, energyList.size());
        m_MaterialMu = MaterialMuImageType::New();
        m_MaterialMu->SetRegions(region);
        m_MaterialMu->Allocate();
        itk::ImageRegionIterator<MaterialMuImageType> it(m_MaterialMu, region);
        for (unsigned int energy = 0; energy < energyList.size(); energy++)
          {
          for (unsigned int i = 0; i < imageWorldMaterials.size(); i++)
            {
            G4Material * mat = imageWorldMaterials[i];
            double mu = 0;
            for (unsigned int process = 0; process < processNameVector.size(); process++)
              {
              /* Note: the G4EmCalculator retrive the correct G4VProcess
               (standard, Penelope, Livermore) from the processName. */
              double crossSection = emCalculator->ComputeCrossSectionPerVolume(energyList[energy],
                                                                               "gamma",
                                                                               processNameVector[process],
                                                                               mat->GetName());
              /* In (length unit)^{-1} according to
               http://www.lcsim.org/software/geant4/doxygen/html/classG4EmCalculator.html#a870d5fffaca35f6e2946da432034bd4c */
              mu += crossSection;
              }
            it.Set(mu);
            ++it;
            }
          }

        muProbe.Stop();
        G4cout
        << "Computation of the mu lookup table took "
        << muProbe.GetTotal() << ' '
        << muProbe.GetUnit()
        << G4endl;

        MaterialMuImageType::SpacingType spacing;
        spacing[0] = 1.;
        if(energyList.size()>1)
          spacing[1] = energyList[1]-energyList[0];
        else
          spacing[1] = 1.*keV;
        m_MaterialMu->SetSpacing(spacing);
        }

      void CreateMaterialDeltaMap(const double energySpacing,
                                  const double energyMax,
                                  GateVImageVolume * gateImageVolume)
        {
        std::vector<double> energyList;
        energyList.push_back(0.);
        while (energyList.back() < energyMax)
          {
          energyList.push_back(energyList.back() + energySpacing);
          }
        CreateMaterialDeltaMap(energyList, gateImageVolume);
        MaterialDeltaImageType::PointType origin;
        origin.Fill(0.);
        m_MaterialDelta->SetOrigin(origin);
        }

      void CreateMaterialDeltaMap(const std::vector<double> & energyList,
                                  GateVImageVolume * gateImageVolume)
        {
        m_EnergyList = energyList;
        itk::TimeProbe deltaProbe;
        deltaProbe.Start();

        /* Get image materials + world */
        std::vector<G4Material*> imageWorldMaterials;
        gateImageVolume->BuildLabelToG4MaterialVector(imageWorldMaterials);
        GateVVolume *volume = gateImageVolume;
        while (volume->GetLogicalVolumeName() != "world_log")
          {
          volume = volume->GetParentVolume();
          }
        imageWorldMaterials.push_back(const_cast<G4Material*>(volume->GetMaterial()));

        MaterialDeltaImageType::RegionType region;
        region.SetSize(0, imageWorldMaterials.size());
        region.SetSize(1, energyList.size());
        m_MaterialDelta = MaterialDeltaImageType::New();
        m_MaterialDelta->SetRegions(region);
        m_MaterialDelta->Allocate();
        itk::ImageRegionIterator<MaterialDeltaImageType> it(m_MaterialDelta, region);
        for (unsigned int energy = 0; energy < energyList.size(); energy++)
          {
          for (unsigned int i = 0; i < imageWorldMaterials.size(); i++)
            {
            G4Material * mat = imageWorldMaterials[i];
            double delta = 0.0;
            double Density = mat->GetDensity() / (g/cm3);
#ifdef GATE_USE_XRAYLIB
#if XRAYLIB_MAJOR > 3
	    xrl_error *error = NULL;
            for (unsigned int i = 0; i < mat->GetElementVector()->size(); ++i)
              {
		delta += (1 - Refractive_Index_Re(mat->GetElementVector()->at(i)->GetSymbol(), energyList[energy]/(keV), 1.0, &error)) * mat->GetFractionVector()[i];
		if (error != NULL) {
		  G4cerr << "error message: " << error->message << "\n";
		  xrl_clear_error(&error); 
		}
              }
#else
            for (unsigned int i = 0; i < mat->GetElementVector()->size(); ++i)
              delta += (1 - Refractive_Index_Re(mat->GetElementVector()->at(i)->GetSymbol(), energyList[energy]/(keV), 1.0)) * mat->GetFractionVector()[i];
#endif	      
#else
            G4Exception( "GateFixedForcedDetectionFunctors::CreateMaterialDeltaMap", "CreateMaterialDeltaMap", FatalException, "Xraylib is not available\n");
#endif
            delta *= Density;
            it.Set(delta);
            ++it;
            }
          }

        deltaProbe.Stop();
        G4cout
        << "Computation of the delta lookup table took "
        << deltaProbe.GetTotal() << ' '
        << deltaProbe.GetUnit()
        << G4endl;

        MaterialDeltaImageType::SpacingType spacing;
        spacing[0] = 1.;
        if(energyList.size()>1)
          spacing[1] = energyList[1]-energyList[0];
        else
          spacing[1] = 1.*keV;
        m_MaterialDelta->SetSpacing(spacing);
        }


      MaterialMuImageType::Pointer GetMaterialMuMap()
        {
        return m_MaterialMu;
        }


      MaterialDeltaImageType::Pointer GetMaterialDeltaMap()
        {
        return m_MaterialDelta;
        }

      double GetIntegralOverDetectorAndReset()
        {
        double result = 0.;
        for (itk::ThreadIdType i = 0; i < ITK_MAX_THREADS; i++)
          {
          result += m_IntegralOverDetector[i];
          m_IntegralOverDetector[i] = 0.;
          }
        return result;
        }

      double GetSquaredIntegralOverDetectorAndReset()
        {
        double result = 0.;
        for (itk::ThreadIdType i = 0; i < ITK_MAX_THREADS; i++)
          {
          result += m_SquaredIntegralOverDetector[i];
          m_SquaredIntegralOverDetector[i] = 0.;
          }
        return result;
        }

      void SetNumberOfPrimaries(G4int i)
        {
        m_NumberOfPrimaries = i;
        }
      void SetResponseDetector(GateEnergyResponseFunctor *_arg)
        {
        m_ResponseDetector = _arg;
        }
      void SetEnergyResolvedParameters(const double bin, const unsigned int slice)
        {
        m_EnergyResolvedBinSize = bin;
        m_EnergyResolvedSliceSize = slice;
        }

      void PreparePhotonsList(const unsigned int & numberOfThreads)
        {
        m_PhotonList.resize(numberOfThreads);
        }
      std::vector<newPhoton> GetPhotonListPerThread(const unsigned int & thread)
        {
        return m_PhotonList[thread];
        }
      std::vector<std::vector<newPhoton> > GetPhotonList()
        {
        return m_PhotonList;
        }
      void ClearPhotonListPerThread(const unsigned int & thread)
        {
        m_PhotonList[thread].clear();
        }

      void ClearPhotonList()
        {
        for (unsigned int thread = 0; thread < m_PhotonList.size(); thread++)
          {
          m_PhotonList[thread].clear();
          }
        }

      void SavePhotonsparameters(const rtk::ThreadIdType threadId,
                                 const VectorType & photonPosition,
                                 const VectorType & photonDirection,
                                 const double & weight,
                                 const double & energy)
        {
        newPhoton photon;
        photon.weight = weight;
        photon.energy = energy;
        for (int i = 0; i < 3; i++)
          {
          photon.position[i] = photonPosition[i];
          photon.direction[i] = photonDirection[i];
          }

        m_PhotonList[threadId].push_back(photon);
        }
      void setGeneratePhotons(const bool & boolean)
        {
        m_generatePhotons = boolean;
        }
      void setGenerateARF(const bool & boolean)
        {
        m_ARF = boolean;
        }

    protected:
      inline void Accumulate(const rtk::ThreadIdType threadId,
                             float & output,
                             const double valueToAccumulate,
                             const double energy)
        {
        if (m_EnergyResolvedBinSize > 0)
          {
          const std::ptrdiff_t offset = m_EnergyResolvedSliceSize
                                        * itk::Math::Floor<unsigned int>(energy
                                                                         / m_EnergyResolvedBinSize
                                                                         + 0.5);
          *(&output + offset) += valueToAccumulate;
          }
        else
          {
          output += valueToAccumulate;
          }
        m_IntegralOverDetector[threadId] += valueToAccumulate;
        m_SquaredIntegralOverDetector[threadId] += valueToAccumulate * valueToAccumulate;
        }

      inline void AccumulatePhase(float & output,
                                  const double valueToAccumulate,
                                  const double energy)
        {
        if (m_EnergyResolvedBinSize > 0)
          {
          const std::ptrdiff_t offset = m_EnergyResolvedSliceSize
                                        * itk::Math::Floor<unsigned int>(energy
                                                                         / m_EnergyResolvedBinSize
                                                                         + 0.5);
          *(&output + offset + m_MuToDeltaImageOffset) += valueToAccumulate;
          }
        else
          {
          *(&output + m_MuToDeltaImageOffset) += valueToAccumulate;
          }
        }

      VectorType m_VolumeSpacing;
      std::vector<double> *m_InterpolationWeights;
      std::vector<double> *m_EnergyWeightList;
      MaterialMuImageType::Pointer m_MaterialMu;
      MaterialDeltaImageType::Pointer m_MaterialDelta;
      VectorType m_DetectorOrientationTimesPixelSurface;
      double m_IntegralOverDetector[ITK_MAX_THREADS];
      double m_SquaredIntegralOverDetector[ITK_MAX_THREADS];
      G4int m_NumberOfPrimaries;
      GateEnergyResponseFunctor *m_ResponseDetector;
      std::vector<double> m_EnergyList;
      std::ptrdiff_t m_MuToDeltaImageOffset;
      double m_EnergyResolvedBinSize;
      unsigned int m_EnergyResolvedSliceSize;
      std::vector<std::vector<newPhoton> > m_PhotonList;
      bool m_generatePhotons;
      bool m_ARF;
      };

    /* Most of the computation for the primary is done in this functor. After a ray
     has been cast, it loops over the energies, computes the ray line integral for
     that energy and takes the exponential of the opposite and add. */
    class PrimaryValueAccumulation: public VAccumulation
      {
    public:

      PrimaryValueAccumulation()
        {
        }
      ~PrimaryValueAccumulation()
        {
        }

      inline void operator()(const rtk::ThreadIdType threadId,
                             const float &itkNotUsed(input),
                             float & output,
                             const double & itkNotUsed(rayCastValue),
                             const VectorType & stepInMM,
                             const VectorType & itkNotUsed(source),
                             const VectorType & pixelToSource,
                             const VectorType & nearestPoint,
                             const VectorType & farthestPoint)
        {
        double *p = m_MaterialMu->GetPixelContainer()->GetBufferPointer();
        /* Multiply interpolation weights by step norm in MM to convert voxel
         intersection length to MM. */
        const double stepInMMNorm = stepInMM.GetNorm();
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size() - 1; j++)
          {
          m_InterpolationWeights[threadId][j] *= stepInMMNorm;
          }

        /* The last material is the world material. One must fill the weight with
         the length from source to nearest point and farthest point to pixel
         point. */
        VectorType worldVector = farthestPoint - nearestPoint - pixelToSource;
        for (int i = 0; i < 3; i++)
          {
          worldVector[i] *= m_VolumeSpacing[i];
          }
        m_InterpolationWeights[threadId].back() += worldVector.GetNorm();

        /* Loops over energy, multiply weights by mu, accumulate using Beer Lambert */
        for (unsigned int i = 0; i < m_EnergyWeightList->size(); i++)
          {
          double rayIntegral = 0.;
          for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size(); j++)
            {
            rayIntegral += m_InterpolationWeights[threadId][j] * *p++;
            }

          /* Statistical noise added */
          if (m_NumberOfPrimaries != 0)
            {
            double a = std::exp(-rayIntegral);
            double nprimE = m_NumberOfPrimaries * (*m_EnergyWeightList)[i];
            double n = ((nprimE)?G4Poisson(nprimE*a)/nprimE:0.);
            this->Accumulate(threadId,
                             output,
                             n * (*m_EnergyWeightList)[i] * (*m_ResponseDetector)(m_EnergyList[i] ),
                             m_EnergyList[i]);
            }
          else
            {
            this->Accumulate(threadId,
                             output,
                             std::exp(-rayIntegral) * (*m_EnergyWeightList)[i],
                             m_EnergyList[i]);
            }

          /* Phase for Fresnel diffraction */
          if (m_MuToDeltaImageOffset != 0)
            {
            double *q = m_MaterialDelta->GetPixelContainer()->GetBufferPointer();
            double rayIntegral = 0.;
            for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size(); j++)
              {
              rayIntegral += m_InterpolationWeights[threadId][j] * *q++;
              }

            /* Matter wave : for photon, the formula is lambda = hc/E
             * https://fr.wikipedia.org/wiki/Hypoth%C3%A8se_de_De_Broglie
             */
            double wavelength = h_Planck/(eV*s) * c_light/(m/s) / (m_EnergyList[i]/eV);
            wavelength = wavelength * (m/mm); // To use the same unit as Geant4
            rayIntegral *= (-2*itk::Math::pi/wavelength);
            this->AccumulatePhase(output,
                                  rayIntegral + std::log((*m_EnergyWeightList)[i])/2.,
                                  m_EnergyList[i]);
            }
          }

        /* Reset weights for next ray in thread.*/
        std::fill(m_InterpolationWeights[threadId].begin(),
                  m_InterpolationWeights[threadId].end(),
                  0.);
        }
      };

    class ComptonValueAccumulation: public VAccumulation
      {
    public:

      ComptonValueAccumulation()
        {
        /* G4 data */
        G4VDataSetAlgorithm* scatterInterpolation = new G4LogLogInterpolation;
        G4String scatterFile = "comp/ce-sf-";
        m_ScatterFunctionData = new G4CompositeEMDataSet(scatterInterpolation, 1., 1.);
        m_ScatterFunctionData->LoadData(scatterFile);

        m_CrossSectionHandler = new G4CrossSectionHandler;
        G4String crossSectionFile = "comp/ce-cs-";
        m_CrossSectionHandler->LoadData(crossSectionFile);
        }
      ~ComptonValueAccumulation()
        {
        delete m_ScatterFunctionData;
        delete m_CrossSectionHandler;
        }

      inline void operator()(const rtk::ThreadIdType threadId,
                             const float &itkNotUsed(input),
                             float &output,
                             const double &itkNotUsed(rayCastValue),
                             const VectorType &stepInMM,
                             const VectorType &itkNotUsed(source),
                             const VectorType &pixelToSource,
                             const VectorType &nearestPoint,
                             const VectorType &farthestPoint)
        {
        /* Compute ray length in world material
         This is used to compute the length in world as well as the direction
         of the ray in mm. */
        VectorType worldVector = farthestPoint - nearestPoint - pixelToSource;
        for (int i = 0; i < 3; i++)
          {
          worldVector[i] *= m_VolumeSpacing[i];
          }
        const double worldVectorNorm = worldVector.GetNorm();

        /* This is taken from G4LivermoreComptonModel.cc */
        double cosT = worldVector * m_Direction / worldVectorNorm;
        double x = std::sqrt(1. - cosT) * m_InvWlPhoton; /* 1-cosT=2*sin(T/2)^2 */
        double scatteringFunction = m_ScatterFunctionData->FindValue(x, m_Z - 1);

        /* This is taken from GateDiffCrossSectionActor.cc and simplified */
        double Eratio = 1. / (1. + m_E0m * (1. - cosT));
        double DCSKleinNishina = m_eRadiusOverCrossSectionTerm
                                 * Eratio
                                 * (1. + Eratio * (Eratio - 1. + cosT * cosT));
        double DCScompton = DCSKleinNishina * scatteringFunction;

        /* Multiply interpolation weights by step norm in MM to convert voxel
         intersection length to MM. */
        const double stepInMMNorm = stepInMM.GetNorm();
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size() - 1; j++)
          {
          m_InterpolationWeights[threadId][j] *= stepInMMNorm;
          }

        /* The last material is the world material. One must fill the weight with
         the length from farthest point to pixel point. */

        if (!m_generatePhotons)
          {
          m_InterpolationWeights[threadId].back() = worldVectorNorm;
          }
        else
          {
          m_InterpolationWeights[threadId].back() = worldVectorNorm;
          }
        const double energy = Eratio * m_Energy;
        unsigned int e = itk::Math::Round<double, double>(energy / m_MaterialMu->GetSpacing()[1]);
        double *p = m_MaterialMu->GetPixelContainer()->GetBufferPointer()
                    + e * m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];

        /* Ray integral */
        double rayIntegral = 0.;
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size(); j++)
          {
          rayIntegral += m_InterpolationWeights[threadId][j] * *p++;
          }

        /* Final computation */
        // double weight = std::exp(-rayIntegral) * DCScompton * GetSolidAngle(sourceToPixel) * (*m_ResponseDetector)(energy);
        double weight = std::exp(-rayIntegral) * DCScompton * GetSolidAngle(pixelToSource);
        if (m_generatePhotons)
          {
          //Accumulate(threadId, output, weight, m_Energy);
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = farthestPoint[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, energy);
          }
        else if (m_ARF)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = -pixelToSource[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, energy);
          }
        else
          {
          Accumulate(threadId, output, weight * (*m_ResponseDetector)(energy), energy);
          }

        /* Reset weights for next ray in thread. */
        std::fill(m_InterpolationWeights[threadId].begin(),
                  m_InterpolationWeights[threadId].end(),
                  0.);
        }

      void SetDirection(const VectorType &_arg)
        {
        m_Direction = _arg;
        }

      void SetEnergyZAndWeight(const double &energy, const unsigned int &Z, const double &weight)
        {
        m_Energy = energy;
        m_Weight = weight;
        m_E0m = m_Energy / electron_mass_c2;
        m_InvWlPhoton = std::sqrt(0.5) * cm * m_Energy / (h_Planck * c_light); /* sqrt(0.5) for trigo reasons, see comment when used */

        G4double crossSection = m_CrossSectionHandler->FindValue(Z, energy);
        m_Z = Z;
        m_eRadiusOverCrossSectionTerm = weight * (classic_electr_radius * classic_electr_radius)
                                        / (2. * crossSection);
        }

    private:
      VectorType m_Direction;
      double m_Energy;
      double m_E0m;
      double m_Weight;

      double m_InvWlPhoton;
      unsigned int m_Z;
      double m_eRadiusOverCrossSectionTerm;
      /* Compton data */
      G4VEMDataSet* m_ScatterFunctionData;
      G4VCrossSectionHandler* m_CrossSectionHandler;
      };

    class RayleighValueAccumulation: public VAccumulation
      {
    public:

      RayleighValueAccumulation()
        {
        /* G4 data */
        m_dataCS.push_back(nullptr);
        m_dataFF.push_back(nullptr);
        for(G4int Z = 0; Z < 100; ++Z)
          {
          m_dataCS.push_back(nullptr);
          ReadData(m_dataCS, Z, "/livermore/rayl/re-cs-");
          m_dataFF.push_back(nullptr);
          ReadData(m_dataFF, Z, "/livermore/rayl/re-ff-");
          }
	      }
      ~RayleighValueAccumulation()
        {
        }

      void ReadData(std::vector<G4PhysicsFreeVector*>& data, size_t Z, const char* path)
        {
        if(nullptr != data[Z]) { return; }

        const char* datadir = nullptr;

        if(datadir == nullptr)
        {
          datadir = std::getenv("G4LEDATA");
          if(datadir == nullptr)
          {
            G4Exception("G4LivermoreRayleighModelModel::ReadData()","em0006",
		        FatalException,
		        "Environment variable G4LEDATA not defined");
            return;
          }
        }
        data[Z] = new G4PhysicsFreeVector();

        std::ostringstream ostCS;
        ostCS << datadir << path << Z+1 <<".dat";

        std::ifstream finCS(ostCS.str().c_str());

        if( !finCS .is_open() )
        {
          G4ExceptionDescription ed;
          ed << "G4LivermoreRayleighModel data file <" << ostCS.str().c_str()
             << "> is not opened!" << G4endl;
          G4Exception("G4LivermoreRayleighModel::ReadData()","em0003",FatalException,
		      ed,"G4LEDATA version should be G4EMLOW6.27 or later.");
          return;
        }
        else
        {
          data[Z]->Retrieve(finCS, true);
        }
        }


      inline void operator()(const rtk::ThreadIdType threadId,
                             const float & itkNotUsed(input),
                             float & output,
                             const double & itkNotUsed(rayCastValue),
                             const VectorType & stepInMM,
                             const VectorType & itkNotUsed(source),
                             const VectorType & pixelToSource,
                             const VectorType & nearestPoint,
                             const VectorType & farthestPoint)
        {
        /* Compute ray length in world material. This is used to compute the length in world as well as the direction of the ray in mm. */
        VectorType worldVector = farthestPoint - nearestPoint - pixelToSource;
        for (int i = 0; i < 3; i++)
          {
          worldVector[i] *= m_VolumeSpacing[i];

          }

        const double worldVectorNorm = worldVector.GetNorm();

        /* This is taken from GateDiffCrossSectionActor.cc and simplified */
        double cosT = worldVector * m_Direction / worldVectorNorm;
        double DCSThomsonTerm1 = (1 + cosT * cosT);
        double DCSThomson = m_eRadiusOverCrossSectionTerm * DCSThomsonTerm1;
        double x = std::sqrt(1. - cosT) * m_InvWlPhoton; /* 1-cosT=2*sin(T/2)^2 */
        G4double formFactor = findValue(m_dataFF, m_Z-1, x);
        double DCSrayleigh = DCSThomson * formFactor * formFactor;

        /* Multiply interpolation weights by step norm in MM to convert voxel
         intersection length to MM. */
        const double stepInMMNorm = stepInMM.GetNorm();
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size() - 1; j++)
          {
          m_InterpolationWeights[threadId][j] *= stepInMMNorm;
          }

        /* The last material is the world material. One must fill the weight with
         the length from farthest point to pixel point. */
        m_InterpolationWeights[threadId].back() = worldVectorNorm;

        /* Ray integral */
        double rayIntegral = 0.;
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size(); j++)
          {
          rayIntegral += m_InterpolationWeights[threadId][j] * *(m_MaterialMuPointer + j);
          }

        /* Final computation */
        double weight = std::exp(-rayIntegral) * DCSrayleigh * GetSolidAngle(pixelToSource);
        if (m_generatePhotons)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = farthestPoint[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, m_Energy);
          }
        else if (m_ARF)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = -pixelToSource[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, m_Energy);
          }

        else
          {
          Accumulate(threadId, output, weight, m_Energy);
          }

        /* Reset weights for next ray in thread. */
        std::fill(m_InterpolationWeights[threadId].begin(),
                  m_InterpolationWeights[threadId].end(),
                  0.);
        }

      void SetDirection(const VectorType &_arg)
        {
        m_Direction = _arg;
        }

      G4double findValue(std::vector<G4PhysicsFreeVector*>& data, const unsigned int & Z, const double & energy)
        {
        G4double xs = 1.0;

        G4PhysicsFreeVector* pv = data[Z];
        G4int n = pv->GetVectorLength() - 1;
        G4double e = energy/MeV;
        if(e >= (*pv).Energy(n)) {
          xs = (*pv)[n];
        } else if(e >= pv->Energy(0)) {
          xs = pv->Value(e);
        }
        return(xs);
        }

      void SetEnergyZAndWeight(const double & energy, const unsigned int & Z, const double & weight)
        {
        unsigned int e = itk::Math::Round<double, double>(energy / m_MaterialMu->GetSpacing()[1]);
        m_InvWlPhoton = std::sqrt(0.5) * cm * energy / (h_Planck * c_light); // sqrt(0.5) for trigo reasons, see comment when used
        m_Energy = energy;
        m_Weight = weight;
        m_MaterialMuPointer = m_MaterialMu->GetPixelContainer()->GetBufferPointer();
        m_MaterialMuPointer += e * m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];

        G4double crossSection = findValue(m_dataCS, Z-1, energy);
        G4double eMeV = energy/MeV;
        crossSection = crossSection/(eMeV*eMeV);
        m_Z = Z;
        m_eRadiusOverCrossSectionTerm = weight * (classic_electr_radius * classic_electr_radius)
                                        / (2. * crossSection);
        }

    private:
      VectorType m_Direction;
      double *m_MaterialMuPointer;
      double m_InvWlPhoton;
      double m_Weight;
      double m_Energy;
      unsigned int m_Z;
      double m_eRadiusOverCrossSectionTerm;

      /* G4 data */
      std::vector<G4PhysicsFreeVector*> m_dataCS;
      std::vector<G4PhysicsFreeVector*> m_dataFF;
      };

    class FluorescenceValueAccumulation: public VAccumulation
      {
    public:

      FluorescenceValueAccumulation()
        {
        }
      ~FluorescenceValueAccumulation()
        {
        }

      inline void operator()(const rtk::ThreadIdType threadId,
                             const float & itkNotUsed(input),
                             float & output,
                             const double &itkNotUsed(rayCastValue),
                             const VectorType & stepInMM,
                             const VectorType & itkNotUsed(source),
                             const VectorType & pixelToSource,
                             const VectorType & nearestPoint,
                             const VectorType & farthestPoint)
        {
        /* Compute ray length in world material
         This is used to compute the length in world as well as the direction
         of the ray in mm. */
        VectorType worldVector = farthestPoint - nearestPoint - pixelToSource;
        for (int i = 0; i < 3; i++)
          {
          worldVector[i] *= m_VolumeSpacing[i];
          }
        const double worldVectorNorm = worldVector.GetNorm();

        /* Multiply interpolation weights by step norm in MM to convert voxel
         intersection length to MM. */
        const double stepInMMNorm = stepInMM.GetNorm();
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size() - 1; j++)
          {
          m_InterpolationWeights[threadId][j] *= stepInMMNorm;
          }

        /* The last material is the world material. One must fill the weight with
         the length from farthest point to pixel point. */
        m_InterpolationWeights[threadId].back() = worldVectorNorm;

        /* Ray integral */
        double rayIntegral = 0.;
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size(); j++)
          {
          rayIntegral += m_InterpolationWeights[threadId][j] * *(m_MaterialMuPointer + j);
          }

        /* Final computation */
        double weight = m_Weight * std::exp(-rayIntegral)*GetSolidAngle(pixelToSource)/(4*itk::Math::pi);
        if (m_generatePhotons)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = farthestPoint[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, m_Energy);
          }
        else if (m_ARF)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = -pixelToSource[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, m_Energy);
          }

        else
          {
          Accumulate(threadId, output, weight, m_Energy);
          }
        /* Reset weights for next ray in thread. */
        std::fill(m_InterpolationWeights[threadId].begin(),
                  m_InterpolationWeights[threadId].end(),
                  0.);

        }

      void SetDirection(const VectorType &itkNotUsed(_arg))
        {
        }
      void SetEnergyZAndWeight(const double &energy,
                               const unsigned int &itkNotUsed(Z),
                               const double &weight)
        {
        unsigned int e = itk::Math::Round<double, double>(energy / m_MaterialMu->GetSpacing()[1]);
        m_Weight = weight;
        m_Energy = energy;
        m_MaterialMuPointer = m_MaterialMu->GetPixelContainer()->GetBufferPointer();
        m_MaterialMuPointer += e * m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];
        }

    private:
      double *m_MaterialMuPointer;
      double m_Weight;
      double m_Energy;
      };

    class IsotropicPrimaryValueAccumulation: public VAccumulation
      {
    public:

      IsotropicPrimaryValueAccumulation()
        {
        }
      ~IsotropicPrimaryValueAccumulation()
        {
        }

      inline void operator()(const rtk::ThreadIdType threadId,
                             const float & itkNotUsed(input),
                             float & output,
                             const double & itkNotUsed(rayCastValue),
                             const VectorType & stepInMM,
                             const VectorType & itkNotUsed(source),
                             const VectorType & pixelToSource,
                             const VectorType & nearestPoint,
                             const VectorType & farthestPoint)
        {
        /* Compute ray length in world material
         This is used to compute the length in world as well as the direction
         of the ray in mm. */
        VectorType worldVector = farthestPoint - nearestPoint - pixelToSource;
        for (int i = 0; i < 3; i++)
          {
          worldVector[i] *= m_VolumeSpacing[i];
          }
        const double worldVectorNorm = worldVector.GetNorm();

        /* Multiply interpolation weights by step norm in MM to convert voxel
         intersection length to MM. */
        const double stepInMMNorm = stepInMM.GetNorm();
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size() - 1; j++)
          {
          m_InterpolationWeights[threadId][j] *= stepInMMNorm;
          }

        /* The last material is the world material. One must fill the weight with
         the length from farthest point to pixel point. */

        if (!m_generatePhotons)
          {
          m_InterpolationWeights[threadId].back() = worldVectorNorm;
          }
        else
          {
          m_InterpolationWeights[threadId].back() = 0;
          }
        /* Ray integral */
        double rayIntegral = 0.;
        for (unsigned int j = 0; j < m_InterpolationWeights[threadId].size(); j++)
          {
          rayIntegral += m_InterpolationWeights[threadId][j] * *(m_MaterialMuPointer + j);
          }

        /* Final computation */
        double weight = m_Weight * std::exp(-rayIntegral)*GetSolidAngle(pixelToSource)/(4*itk::Math::pi);
        if (m_generatePhotons)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = farthestPoint[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, m_Energy);
          }
        else if (m_ARF)
          {
          VectorType photonDirection;
          VectorType photonPosition;
          for (int i = 0; i < 3; i++)
            {
            photonDirection[i] = worldVector[i] / worldVectorNorm;
            photonPosition[i] = -pixelToSource[i] * m_VolumeSpacing[i];
            }
          SavePhotonsparameters(threadId, photonPosition, photonDirection, weight, m_Energy);
          }

        else
          {
          Accumulate(threadId, output, weight, m_Energy);
          }

        /* Reset weights for next ray in thread. */
        std::fill(m_InterpolationWeights[threadId].begin(),
                  m_InterpolationWeights[threadId].end(),
                  0.);
        }

      void SetDirection(const VectorType &itkNotUsed(_arg))
        {
        }
      void SetEnergyZAndWeight(const double & energy,
                               const unsigned int &itkNotUsed(Z),
                               const double & weight)
        {
        unsigned int e = itk::Math::Round<double, double>(energy / m_MaterialMu->GetSpacing()[1]);
        m_Weight = weight;
        m_Energy = energy;
        m_MaterialMuPointer = m_MaterialMu->GetPixelContainer()->GetBufferPointer();
        m_MaterialMuPointer += e * m_MaterialMu->GetLargestPossibleRegion().GetSize()[0];
        }

    private:
      double *m_MaterialMuPointer;
      double m_Weight;
      double m_Energy;
      };

    template<class TInput1, class TInput2 = TInput1, class TOutput = TInput1>
    class Attenuation
      {
    public:
      Attenuation()
        {
        }
      ~Attenuation()
        {
        }
      bool operator!=(const Attenuation &) const
        {
        return false;
        }

      bool operator==(const Attenuation & other) const
        {
        return !(*this != other);
        }

      inline TOutput operator()(const TInput1 A, const TInput2 B) const
        {
        /* Calculating attenuation image (-log(primaryImage/flatFieldImage)) */
        return (TOutput) (-log(A / B));
        }
      };

    template<class TInput1, class TInput2 = TInput1, class TOutput = TInput1>
    class Chetty
      {
    public:
      Chetty()
        {
        }
      ~Chetty()
        {
        }
      bool operator!=(const Chetty &) const
        {
        return false;
        }

      bool operator==(const Chetty & other) const
        {
        return !(*this != other);
        }

      void SetN(double N)
        {
        m_invN = 1 / N;
        m_invNm1 = 1 / (N - 1);
        }

      inline TOutput operator()(const TInput1 sum, const TInput2 squaredSum) const
        {
        /* Chetty, IJROBP, 2006, p1250, eq 2 */
        return sqrt(m_invNm1 * (squaredSum * m_invN - pow(sum * m_invN, 2.))) / (sum * m_invN);
        }
    private:
      double m_invN;
      double m_invNm1;
      };

    template<class TInput1, class TInput2 = TInput1, class TOutput = std::complex<TInput1>>
    class Transmittance
      {
    public:
      Transmittance()
        {
        }
      ~Transmittance()
        {
        }
      bool operator!=(const Transmittance &) const
        {
        return false;
        }

      bool operator==(const Transmittance & other) const
        {
        return !(*this != other);
        }

      inline TOutput operator()(const TInput1 A, const TInput2 B) const
        {
        /* Calculate transmittance */
        std::complex<TInput1> amp = std::complex<TInput1>(std::sqrt(A), 0.);  // Amplitude
        std::complex<TInput1> pt = std::exp(std::complex<TInput1>(0., B));  // Phase Transition
        return (TOutput) (amp*pt);
        }
      };

  }

#endif // GATEFIXEDFORCEDDETECTIONACTORFUNCTORS_HH
