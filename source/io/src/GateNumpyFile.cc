//
// Created by mdupont on 30/08/17.
//

#include "GateNumpyFile.hh"
#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip>
#include <cstring>

#include <stdexcept>
#include <utility>
#include "GateFileExceptions.hh"
#include "GateTreeFileManager.hh"

#include "GateMessageManager.hh"

using namespace std;
//
//
int is_big_endian()
{
  union {
    long int l;
    char c[sizeof (long int)];
  } u{};

  u.l = 1;

  if (u.c[sizeof(long int)-1] == 1)
    {
      return 1;
    }
  else
    return 0;
}

string get_endianness_carater()
{
  if (is_big_endian())
    return ">";
  return "<";
}
//

GateNumpyTree::GateNumpyTree() : m_nb_elements(0)
{
  //  cout << "NumpyFile::NumpyFile()\n";
  //  cout << "NumpyFile::NumpyFile() is_big_endian = " << is_big_endian() << "\n";

  fill_maps<long double>("f16");
  fill_maps<double>("f8");
  fill_maps<float>("f4");
  fill_maps<uint8_t>("u1");
  fill_maps<uint16_t>("u2");
  fill_maps<uint32_t>("u4");
  fill_maps<uint64_t>("u8");
  fill_maps<int8_t>("i1");
  fill_maps<int16_t>("i2");
  fill_maps<int32_t>("i4");
  fill_maps<int64_t>("i8");

  fill_maps<bool>("?");
  fill_maps<char>("b");

  //  m_file = ofstream();

  //  m_file.open("/tmp/z.npy",std::ofstream::binary);
}


void GateNumpyTree::fill_maps(std::type_index t_index, const std::string& s)
{
  m_tmap_cppToNumpy.emplace(t_index, get_endianness_carater() + s);
  m_tmap_numpyToCpp.emplace(get_endianness_carater() + s, t_index);
}


void GateNumpyTree::register_variable(const std::string& name, const void *p, const size_t size, std::string numpy_format, std::type_index t_index)
{

  GateNumpyData d(p, size, name, std::move(numpy_format), t_index);

  for (auto&& d_ : m_vector_of_pointer_to_data) // access by const reference
    {
      if(d_.name() == name)
        {
          string s("Error: Key '");
          s+= name;
          s += "' already used !";
          cerr << s << endl;
          throw GateKeyAlreadyExistsException( s );
        }
    }
  m_vector_of_pointer_to_data.push_back(d);
}
//
void GateNumpyTree::register_variable(const std::string &name, const char *p, size_t nb_char)
{

  if(!nb_char)
    throw std::out_of_range("nb_char == 0 does not make any sense");

  stringstream ss;
  ss << "|S" << nb_char;
  register_variable(name, p, sizeof(char)*nb_char, ss.str(), typeid(char*));
  auto&& data = m_vector_of_pointer_to_data.back();
  data.m_nb_characters = nb_char;
}
//
//
void GateNumpyTree::register_variable(const std::string &name, const std::string *p, size_t nb_char)
{
  if(!nb_char)
    throw std::out_of_range("nb_char == 0 does not make any sense");

  stringstream ss;
  ss << "|S" << nb_char;
  register_variable(name, p, sizeof(char)*nb_char, ss.str(), typeid(string));
  auto&& data = m_vector_of_pointer_to_data.back();
  data.m_nb_characters = nb_char;
}


void  GateNumpyTree::register_variable(const std::string &name, const int *, size_t n)
{
    //Thinking about how to write in npy file an array of int corresponding to volumeID information. Work in progres. It is not working properly

    if(!n)
        throw std::out_of_range("n == 0 does not make any sense");
     std::cout<<"ERROR "<<name<<" information can not be written in .npy output file"<<std::endl;
     GateError(".npy format  is not available for *int output such as  volumeID. Unselect volumeID information in your output file or select .root output file");
    //stringstream ss;
    //ss << "|V" << n;
    //register_variable(name, p, sizeof(int)*n, ss.str(), typeid(int*));
    //auto&& data = m_vector_of_pointer_to_data.back();
    //data.m_nb_characters = n;
}



void GateOutputNumpyTreeFile::write_header()
{

  if( (m_mode & ios_base::out) != ios_base::out )
    throw std::runtime_error("NumpyFile::write_header: file not opened in write mode");

  m_file << magic_prefix.c_str();
  uint32_t  magic_len = magic_prefix.size() + 2;

  std::stringstream ss_dico_before_shape, ss_dico_after_shape;
  string dico_before_shape, dico_after_shape;

  ss_dico_before_shape << "{'descr': [";


  for ( auto it = m_vector_of_pointer_to_data.begin(); it != m_vector_of_pointer_to_data.end(); ++it )
    {
      if(it != m_vector_of_pointer_to_data.begin())
        ss_dico_before_shape << ", ";
      ss_dico_before_shape << (*it).m_numpy_description;
    }

  ss_dico_before_shape << "], 'fortran_order': False, 'shape': (";

  string shape = "######";

  stringstream ss_shape;
  ss_shape << std::setw(20) << std::setfill(' ') << 0;
  shape = ss_shape.str();

  ss_dico_after_shape << ",),}";


  dico_before_shape = ss_dico_before_shape.str();
  dico_after_shape = ss_dico_after_shape.str();


  uint32_t  current_header_len = magic_len + 2  + dico_before_shape.size() + shape.size() + dico_after_shape.size() + 1;  // 1 for the newline
  uint32_t  topad = 16 - (current_header_len % 16);

  for(uint32_t i = 0; i < topad; ++i)
    dico_after_shape += ' ';

  dico_after_shape += '\n';

  uint16_t hlen = dico_before_shape.size() + shape.size() + dico_after_shape.size();

  uint8_t major = 1;
  uint8_t minor = 0;

  m_file.write((char*)&major, sizeof(major));
  m_file.write((char*)&minor, sizeof(minor));
  m_file.write((char*)&hlen, sizeof(hlen));


  m_file << dico_before_shape.c_str();
  m_position_before_shape = m_file.tellp();

  m_file.write(shape.c_str(), shape.size());
  m_position_after_shape = m_file.tellp();
  m_file << dico_after_shape.c_str();
  m_write_header_called = true;
}


void GateOutputNumpyTreeFile::fill()
{

  if(!m_write_header_called)
    throw std::logic_error("write_header not called");

  if (m_vector_of_pointer_to_data.empty())
    return;

  //  if( (m_mode & ios_base::out) != ios_base::out )
  //    return;

  //  cout << "OutputNumpyTreeFile::fill()" << endl;
  for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
    {
      if(d.m_nb_characters == 0)
        m_file.write((const char*)d.m_pointer_to_data, d.m_size_of_data);
      else
        {
          if(d.m_type_index == typeid(char*))
            {
              char *p_data = (char*)d.m_pointer_to_data;
              auto current_nb_characters = strlen(p_data);
              //        cout << " p_data = " << p_data;
              //        cout << " current_nb_characters = " << current_nb_characters;
              for(size_t i = current_nb_characters; i < d.m_nb_characters; ++i)
                p_data[i] = '\0';


              m_file.write(p_data, d.m_size_of_data);
          } else if (d.m_type_index == typeid(string))
          {
              const auto *p_s = (const string*) d.m_pointer_to_data;
              if( p_s->size() > d.m_nb_characters)
              {
                  string m;
                  m += "length(" + *p_s + ") = (" + std::to_string(p_s->size()) +   ") > " + std::to_string(d.m_nb_characters);
                  throw std::length_error(m);
              }

              string s(*p_s); // copy of the data, :-(
              s.resize(d.m_nb_characters, '\0');


              m_file.write(s.c_str(), d.m_size_of_data);
          }
          /*
           *Thinking about how to write in npy file an array of int corresponding to volumeID information. Work in progres. It is not working
           * else if (d.m_type_index == typeid(int*))
          {
              int *p_int = (int*) d.m_pointer_to_data;
              std::string numericArrayToWrite ="[";
              for (auto numericIndex=0; numericIndex<d.m_nb_characters-1; ++numericIndex)
                  numericArrayToWrite += std::to_string(p_int[numericIndex]) + ",";
              numericArrayToWrite += std::to_string(p_int[d.m_nb_characters-1]) + "]";
              m_file.write(numericArrayToWrite.c_str(), d.m_size_of_data+2+d.m_nb_characters-1);
          }*/
        }
    }

  m_nb_elements++;
  //  cout << "\n";
}

void GateOutputNumpyTreeFile::write()
{

  if(!m_file.is_open())
    return;

  if( (m_mode & ios_base::out) == ios_base::out )
    {
      m_file.seekp(m_position_before_shape);
      //    cout << "current position = " << m_file.tellp() << "\n";
      stringstream ss_shape;
      ss_shape << std::setw(20) << std::setfill(' ') << m_nb_elements;
      string shape = ss_shape.str();
      m_file.write(shape.c_str(), shape.size());
    } else {
    for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
      {
        if(d.buffer_read)
          delete[](d.buffer_read);
      }
    //    throw std::runtime_error("NumpyTreeFile::close: no se");
  }
}

void GateOutputNumpyTreeFile::close()
{

  if(!m_file.is_open())
    return;

  GateOutputNumpyTreeFile::write();

  m_file.close();
}

void GateInputNumpyTreeFile::close()
{

  if(!m_file.is_open())
    return;

  m_file.close();
}



void GateOutputNumpyTreeFile::open(const std::string& s)
{
  GateFile::open(s, std::ofstream::binary | std::fstream::out);
  m_file.open(s, std::ofstream::binary | std::fstream::out);
  if(!m_file.is_open())
    {
      //    perror("Error opening file!");
      std::stringstream ss;
      ss << "Error opening file! '"  << s <<  "' : " << strerror(errno) ;
      throw std::ios::failure(ss.str());
    }
}

void GateInputNumpyTreeFile::open(const std::string &s)
{
  GateFile::open(s, std::fstream::in);
  m_file.open(s, std::ofstream::binary | std::fstream::in);
  if(!m_file.is_open())
    {
      //    perror("Error opening file!");
      std::stringstream ss;
      ss << "Error opening file! '"  << s <<  "' : " << strerror(errno) ;
      throw std::ios::failure(ss.str());
    }

  // get length of file:
  m_file.seekg (0, std::fstream::end);
  m_length_of_file = m_file.tellg();
  m_file.seekg (0, std::fstream::beg);
}

void GateOutputNumpyTreeFile::write_variable(const std::string &name, const void *p, std::type_index t_index)
{
  this->register_variable(name, p, t_index);
}

void GateOutputNumpyTreeFile::write_variable(const std::string &name, const std::string *p, size_t nb_char)
{
  this->register_variable(name, p, nb_char);
}

void GateOutputNumpyTreeFile::write_variable(const std::string &name, const char *p, size_t nb_char)
{
  this->register_variable(name, p, nb_char);
}

void GateOutputNumpyTreeFile::write_variable(const std::string &name, const int *p, size_t nb)
{
  this->register_variable(name, p, nb);
}

GateOutputNumpyTreeFile::GateOutputNumpyTreeFile() : m_write_header_called(false)
{}





//
void GateNumpyTree::register_variable(const std::string &name, const void *p, std::type_index t_index)
{
  //  std::cout << "NumpyFile::register_variable name = " << name << " t_index = " << t_index.name() << "\n";
  this->register_variable(name, p, m_tmapOfSize.at(t_index), m_tmap_cppToNumpy.at(t_index), t_index);
}

bool GateOutputNumpyTreeFile::is_open()
{
  return m_file.is_open();
}



bool GateInputNumpyTreeFile::is_open()
{
  return m_file.is_open();
}

uint64_t GateNumpyTree::nb_elements()
{
  return m_nb_elements;
}


string get_data(const string &str)
{
  size_t pos = str.rfind(',');
  string s = str.substr(0, pos);
  pos = s.find(':') + 1;
  return s.substr( pos, s.size() - pos  );
}


vector<pair<string, string>> intertpret_descr(const string &descr)
{
  //  cout << "intertpret_descr:#" << descr << "#" << "\n";
  string str = descr;

  size_t pos_start = 0;
  size_t pos_end = descr.size();

  pos_start = str.find('(');
  pos_end = str.find(')');

  std::vector<string> vect_data;

  while( pos_start != string::npos  )
    {


      //cout << "pos_start=" << pos_start << " " << "pos_end=" <<  pos_end << "\n";

      string data = str.substr(pos_start, pos_end - 1);

      str = str.substr(pos_end + 1, str.size() - pos_end -1);


      //    cout << "str=" << str << "\n";
      //    cout << "data = " << data << "\n";
      vect_data.push_back(data);

      pos_start = str.find('(');
      pos_end = str.find(')');
    }

  if (vect_data.empty())
    throw std::runtime_error("only structured data accepted");


  vector<pair<string, string>> v;

  for (auto&& data : vect_data)
    {

      size_t pos = data.find(',');
      string name = data.substr( 0, pos );
      string type = data.substr( pos + 1, data.size() - pos -1);

      pos_start = name.find("'");
      pos_end = name.rfind("'");
      name = name.substr(pos_start + 1, pos_end - pos_start - 1);

      pos_start = type.find("'");
      pos_end = type.rfind("'");
      type = type.substr(pos_start + 1, pos_end - pos_start - 1);

      //    cout << "for data=" << data << " name = #" << name << "# type = " << type << " \n";

      v.emplace_back(name, type);

    }

  return v;


  //  return vect_data;
}

uint64_t interpret_shape(const string &shape)
{
  size_t pos_start = shape.find('(');
  size_t comma = shape.find(',');
  string data = shape.substr(pos_start + 1, comma - 1);
  return stol(data);
}

void GateInputNumpyTreeFile::read_header()
{

  if(!m_file.is_open())
    throw std::runtime_error("InputNumpyTreeFile::read_header: try to read from closed file");


  string magic(6, '\0');
  m_file.read(&magic[0], 6);

  bool e = magic == "\x93NUMPY";
  if(!e)
    throw std::runtime_error("InputNumpyTreeFile::read_header: failed to find magic numpy word");

  uint8_t major = 0;
  uint8_t minor = 0;

  m_file.read((char*)&major, sizeof(major));
  m_file.read((char*)&minor, sizeof(minor));

  if(major != 1)
    throw std::runtime_error("InputNumpyTreeFile::read_header: major != 1");

  if(minor != 0)
    throw std::runtime_error("InputNumpyTreeFile::read_header: minor != 0");

  uint16_t hlen = 0;
  m_file.read((char*)&hlen, sizeof(hlen));

  //  cout << "hlen = " << hlen << "\n";

  if(hlen == 0)
    throw std::runtime_error("InputNumpyTreeFile::read_header: hlean == 0");

  //  cout << "hlen = " << hlen << "\n";

  string dictionary(hlen, '\0');
  m_file.read(&dictionary[0], hlen);

  //  cout << "dictionary=#" << dictionary << "#" << "\n";

  size_t fortran_order_pos = dictionary.find("'fortran_order'");
  size_t descr_pos = dictionary.find("'descr'");
  size_t shape_pos = dictionary.find("'shape'");




  //  cout << "descr_pos = " << descr_pos << "\n";
  //  cout << "fortran_order_pos = " << fortran_order_pos << "\n";
  //
  //  cout << "shape_pos = " << shape_pos << "\n";

  if(  descr_pos > fortran_order_pos or fortran_order_pos > shape_pos )
    throw std::runtime_error("InputNumpyTreeFile::read_header: wrong order of key in description");

  string descr_str = dictionary.substr(descr_pos, fortran_order_pos - descr_pos);
  //  cout << "descr_str = #" << descr_str << "#\n";
  string fortran_order_str = dictionary.substr(fortran_order_pos, shape_pos - fortran_order_pos);
  //  cout << "fortran_order_str = #" << fortran_order_str << "#\n";

  string shape_str = dictionary.substr(shape_pos,  dictionary.size() - shape_pos);
  //  cout << "shape_str = #" << shape_str << "#\n";



  descr_str = get_data(descr_str);
  fortran_order_str = get_data(fortran_order_str);
  shape_str = get_data(shape_str);

  m_nb_elements = interpret_shape(shape_str);


  auto  v_descr_str = intertpret_descr(descr_str);

  for (auto&& data : v_descr_str)
    {
      auto name = data.first;
      auto type_str = data.second;

      //    cout << "InputNumpyTreeFile::read_header:name = " << name << "\ttype = " << type_str<<  "\n";

      if(  type_str.find("|S") != string::npos  ) // string
        {
          size_t number_caracters = stol(  type_str.substr(2, type_str.size()) );
          this->register_variable(name, (const string *)nullptr, number_caracters);
        }
      else
        {
          auto type_index = m_tmap_numpyToCpp.at(type_str);
          this->register_variable(name, nullptr, type_index);
        }
    }
  m_start_of_data = m_file.tellg();
  m_read_header_called = true;
}

void GateInputNumpyTreeFile::read_next_entrie()
{

  if(!m_read_header_called)
    throw std::logic_error("read_header not called");

  //  cout << "0. current pos = " << m_file.tellg() << " end = " << m_file.end << "eof = " <<  m_file.eof() << "data_to_read() ="  << data_to_read() <<   "\n";
  for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
    {
      //    cout << "InputNumpyTreeFile::read_next_entrie = " << d.m_name << "\ttype = " << d.m_numpy_format<<  "\n";

      if(d.m_pointer_to_data)
        {
          if(!d.m_nb_characters)
            {
              //        cout << "read data for " << d.name() << "\n";
              m_file.read((char*)d.m_pointer_to_data,  d.m_size_of_data);
            }
          else
            {
              if( d.m_type_index_read == typeid(string) )
                {
                  //          char buffer[d.m_nb_characters];
                  m_file.read(d.buffer_read,  d.m_size_of_data);
                  ((string*)d.m_pointer_to_data)->assign(d.buffer_read);
                }
              else
                m_file.read((char*)d.m_pointer_to_data,  d.m_size_of_data);
            }

        }
      else
        {
          m_file.seekg((size_t )m_file.tellg() + d.m_size_of_data);
        }
    }

  //  cout << "1. current pos = " << m_file.tellg() << " end = " << m_file.end << "eof = " <<  m_file.eof() << "data_to_read()"   << data_to_read() <<   "\n";

}

void GateInputNumpyTreeFile::read_variable(const std::string &name, void *p, std::type_index t_index)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");

  for (auto&& d : m_vector_of_pointer_to_data)
    {
      if(name == d.name())
        {
          if(t_index == typeid(string))
            {

              if(d.m_type_index != typeid(string))
                throw GateTypeMismatchHeaderException("Provided a string for non string variable");

              d.m_type_index_read = t_index;
              d.buffer_read = new char[d.m_nb_characters];
              d.m_pointer_to_data = p;
              return;
            }

          if(t_index == typeid(char*))
            {
              if(d.m_type_index != typeid(string))
                throw GateTypeMismatchHeaderException("Provided a char* for non string variable");
              d.m_type_index_read = t_index;
              d.m_pointer_to_data = p;
              return;
            }


          if(t_index != d.m_type_index)
            {

              std::stringstream ss;
              if (d.m_type_index != typeid(string)) // no typeid(char*) in m_tmap_cppToNumpy
                ss << "type_index given to store '" << name <<"' has not the right type (given " << m_tmap_cppToNumpy.at(t_index) << " but need " << m_tmap_cppToNumpy.at(d.m_type_index) << ")";
              else
                {
                  ss << "TODO: make a message here";
                }

              throw GateTypeMismatchHeaderException(ss.str());
            }


          if(this->m_tmapOfSize[t_index] !=  this->m_tmapOfSize[d.m_type_index] )
            {
              std::stringstream ss;
              ss << "object given to store '" << name <<"' has not the right size (given " << this->m_tmapOfSize[t_index] <<" but need " << this->m_tmapOfSize[d.m_type_index] << ")";
              throw GateTypeMismatchHeaderException(ss.str());
            }

          d.m_pointer_to_data = p;
          d.m_type_index_read = t_index;
          return;
        }

    }
  {
    std::stringstream ss;
    ss << "Variable named '" << name << "' not found !";
    throw GateKeyNotFoundInHeaderException(ss.str());
  }


}

GateInputNumpyTreeFile::GateInputNumpyTreeFile() : m_read_header_called(false)
{}

void GateInputNumpyTreeFile::read_variable(const std::string &name, char *p)
{
  read_variable(name, p, typeid(char*));
}

bool GateInputNumpyTreeFile::data_to_read()
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");
  return m_length_of_file > (size_t)m_file.tellg(); // cast to remove warning
}

void GateInputNumpyTreeFile::read_variable(const std::string &name, std::string *p)
{
  read_variable(name, p, typeid(string));
}


bool GateInputNumpyTreeFile::has_variable(const std::string &name)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");
  for (auto&& d : m_vector_of_pointer_to_data) {
    if (name == d.name())
      return true;
  }
  return false;
}

uint64_t GateInputNumpyTreeFile::nb_elements()
{
  return GateNumpyTree::nb_elements();
}

void GateInputNumpyTreeFile::read_entrie(const uint64_t &i)
{
  //  m_file.seekg(m_start_of_data);

  size_t one_element = 0;
  for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
    {
      one_element += d.m_size_of_data;
    }

  m_file.seekg(m_start_of_data + i * one_element);
  this->read_next_entrie();
  return;


  for(uint64_t nb_seek = 0; nb_seek < i; ++nb_seek)
    {
      for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
        {
          m_file.seekg((size_t )m_file.tellg() + d.m_size_of_data);
        }
    }
  this->read_next_entrie();
}

type_index GateInputNumpyTreeFile::get_type_of_variable(const std::string &name)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");
  for (auto&& d : m_vector_of_pointer_to_data) {
    if (name == d.name()) {
      return d.m_type_index;
    }
  }
  std::stringstream ss;
  ss << "Variable named '" << name << "' not found !";
  throw GateKeyNotFoundInHeaderException(ss.str());
}


bool GateOutputNumpyTreeFile::s_registered =  GateOutputTreeFileFactory::_register(GateOutputNumpyTreeFile::_get_factory_name(), &GateOutputNumpyTreeFile::_create_method<GateOutputNumpyTreeFile>);
bool GateInputNumpyTreeFile::s_registered =  GateInputTreeFileFactory::_register(GateInputNumpyTreeFile::_get_factory_name(), &GateInputNumpyTreeFile::_create_method<GateInputNumpyTreeFile>);









