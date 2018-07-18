/*----------------------
 GATE version name: gate_v6

 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

/* Gate */
#include "GateSingleFixedForcedDetectionActor.hh"

/* ITK */
#include <itkImageFileWriter.h>

/* RTK */
#include <rtkMacro.h>

/* Constructors */
GateSingleFixedForcedDetectionActor::GateSingleFixedForcedDetectionActor(G4String name, G4int depth) :
    GateFixedForcedDetectionActor(name, depth)
  {
  GateDebugMessageInc("Actor",4,"GateSingleFixedForcedDetectionActor() -- begin"<<G4endl);
  pActorMessenger = new GateSingleFixedForcedDetectionActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateSingleFixedForcedDetectionActor() -- end"<<G4endl);
  }

/* Destructor */
GateSingleFixedForcedDetectionActor::~GateSingleFixedForcedDetectionActor()
  {
  delete pActorMessenger;
  }

/* Construct */
void GateSingleFixedForcedDetectionActor::Construct()
  {
  GateFixedForcedDetectionActor::Construct();
  //  Callbacks
  EnableBeginOfRunAction(true);
  EnableEndOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnableEndOfEventAction(false);
  //   EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(false);
  }

/* Callback Begin of Run */
void GateSingleFixedForcedDetectionActor::BeginOfRunAction(const G4Run*r)
  {
  GateFixedForcedDetectionActor::BeginOfRunAction(r);

  /* Create a single event if asked for it */
  if (mSingleInteractionFilename != "")
    {
    mInteractionPosition = mSingleInteractionPosition;
    mInteractionDirection = mSingleInteractionDirection;
    mInteractionEnergy = mSingleInteractionEnergy;
    mInteractionWeight = 1.;
    mInteractionZ = mSingleInteractionZ;
    mSingleInteractionImage = CreateVoidProjectionImage();
    if (mMapProcessNameWithType.find(mSingleInteractionType) == mMapProcessNameWithType.end())
      {
      GateWarning("Unhandled gamma interaction in GateFixedForcedDetectionActor / single interaction. Process name is " << mSingleInteractionType << ".\n");
      }
    else
      {
      switch (mMapProcessNameWithType[mSingleInteractionType])
        {
        case COMPTON:
          this->ForceDetectionOfInteraction<COMPTON>(mComptonProjector.GetPointer(),
                                                     mSingleInteractionImage);
          break;
        case RAYLEIGH:
          mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy) * mInteractionWeight;
          this->ForceDetectionOfInteraction<RAYLEIGH>(mRayleighProjector.GetPointer(),
                                                      mSingleInteractionImage);
          break;
        case PHOTOELECTRIC:
          mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy) * mInteractionWeight;
          this->ForceDetectionOfInteraction<PHOTOELECTRIC>(mFluorescenceProjector.GetPointer(),
                                                           mSingleInteractionImage);
          break;
        case ISOTROPICPRIMARY:
          mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy) * mInteractionWeight;
          this->ForceDetectionOfInteraction<ISOTROPICPRIMARY>(mIsotropicPrimaryProjector.GetPointer(),
                                                              mSingleInteractionImage);
          break;
        default:
          GateError("Implementation problem, unexpected process type reached.");
        }
      }

    itk::ImageFileWriter<InputImageType>::Pointer imgWriter;
    imgWriter = itk::ImageFileWriter<InputImageType>::New();
    if (mSingleInteractionFilename != "")
      {
      imgWriter->SetFileName(mSingleInteractionFilename);
      imgWriter->SetInput(mSingleInteractionImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }
    }
  }

#endif
