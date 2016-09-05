#ifndef GATEFIXEDFORCEDDETECTIONPROJECTOR_HH
#define GATEFIXEDFORCEDDETECTIONPROJECTOR_HH

#include <rtkJosephForwardProjectionImageFilter.h>
#include "GateFixedForcedDetectionFunctors.hh"

template<class TProjectedValueAccumulation>
class ITK_EXPORT GateFixedForcedDetectionProjector: public rtk::JosephForwardProjectionImageFilter<
    itk::Image<float, 3>, itk::Image<float, 3>,
    GateFixedForcedDetectionFunctor::InterpolationWeightMultiplication, TProjectedValueAccumulation>
  {
public:
  /** Standard class typedefs. */
  typedef GateFixedForcedDetectionProjector Self;
  typedef rtk::JosephForwardProjectionImageFilter<itk::Image<float, 3>, itk::Image<float, 3>,
      GateFixedForcedDetectionFunctor::InterpolationWeightMultiplication,
      TProjectedValueAccumulation> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Other typedefs. */
  typedef itk::Image<float, 3> ImageType;
  typedef ImageType::Pointer ImagePointer;
  typedef itk::ImageRegionIterator<ImageType> RegionIteratorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)
  ;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GateFixedForcedDetectionProjector, JosephForwardProjectionImageFilter)
  ;

protected:
  GateFixedForcedDetectionProjector()
    {
    }
  virtual ~GateFixedForcedDetectionProjector()
    {
    }

private:
  GateFixedForcedDetectionProjector(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented
  };

#endif
