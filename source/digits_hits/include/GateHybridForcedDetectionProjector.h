#ifndef GATEHYBRIDFORCEDDETECTIONPROJECTOR_HH
#define GATEHYBRIDFORCEDDETECTIONPROJECTOR_HH

#include <rtkJosephForwardProjectionImageFilter.h>
#include "GateHybridForcedDetectionFunctors.hh"

template < class TProjectedValueAccumulation >
class ITK_EXPORT GateHybridForcedDetectionProjector :
 public rtk::JosephForwardProjectionImageFilter < itk::Image<float,3>,
                                                  itk::Image<float,3>,
                                                  GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
                                                  TProjectedValueAccumulation >
{
public:
  /** Standard class typedefs. */
  typedef GateHybridForcedDetectionProjector                     Self;
  typedef  rtk::JosephForwardProjectionImageFilter <
    itk::Image<float,3>,
    itk::Image<float,3>,
    GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
    TProjectedValueAccumulation >                                Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;

  /** Other typedefs. */
  typedef itk::Image<float,3>                                    ImageType;
  typedef ImageType::Pointer                                     ImagePointer;
  typedef itk::ImageRegionIterator< ImageType >                  RegionIteratorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GateHybridForcedDetectionProjector, JosephForwardProjectionImageFilter);

protected:
  GateHybridForcedDetectionProjector() {}
  virtual ~GateHybridForcedDetectionProjector() {}

private:
  GateHybridForcedDetectionProjector(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented
};

#endif
