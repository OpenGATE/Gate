#include "metaObject.h"
#include "metaImage.h"
#include <iostream>

int main(int /*argc*/, char **argv)
{
  int m_SubSamplingFactor = 1;
  MetaImage m_MetaImage;
  if(!m_MetaImage.Read(argv[1], false))
    {
    std::cerr << "File cannot be read: "
             << argv[1] << " for reading."
             << std::endl;
    }
  for(int i=0; i<m_MetaImage.NDims(); i++)
    {
    std::cout << "Dimension #" << i << "=" << m_MetaImage.DimSize(i)/m_SubSamplingFactor << std::endl;
    std::cout << "Dimension #" << i << "=" << m_MetaImage.ElementSpacing(i)*m_SubSamplingFactor << std::endl;
    std::cout << "Origin #" << i << "=" << m_MetaImage.Position(i) << std::endl;
    } 
  return 0;
}
