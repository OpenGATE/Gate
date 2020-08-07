//
// Created by mdupont on 13/03/18.
//

#pragma once


#include "GateTreeFile.hh"
#include "GateTreeFileManager.hh"


class TFile;
class TTree;

class GateRootTree : public GateTree
{
public:
  GateRootTree();
  static std::string _get_factory_name() { return "root"; }
  void register_variable(const std::string &name, const void *p, std::type_index t_index) override;
  void register_variable(const std::string &name, const std::string *p, size_t nb_char) override;
  void register_variable(const std::string &name, const char *p, size_t nb_char) override;
  void register_variable(const std::string &name, const int *p, size_t n) override;

 protected:
  template<typename T>
  void register_variable(const std::string &name, const T *p)
  {
      register_variable(name, p, typeid(T));
  }

  virtual void close();

  TFile * m_file;
  TTree *m_ttree;
private:


  template<typename T>
  void add_leaflist(const std::string &key, const std::string &long_key)
  {
    m_tmapOfDefinition.emplace(typeid(T), key);
    m_tmapOfTindexToLongDefinition.emplace(typeid(T), long_key);
    if(std::is_fundamental<T>::value)
      m_tmapOfLongDefinitionToIndex.emplace(long_key, typeid(T));
  }

protected:
  std::unordered_map<std::type_index, std::string> m_tmapOfDefinition;
  std::unordered_map<std::type_index, std::string> m_tmapOfTindexToLongDefinition;
  std::unordered_map<std::string, std::type_index> m_tmapOfLongDefinitionToIndex;


};



class GateOutputRootTreeFile: public GateRootTree, public GateOutputTreeFile
{
 public:
  void open(const std::string& s) override ;

  bool is_open() override ;
  void close() override ;

  void write_header() override ;
  void write() override ;
  virtual void fill() override;


  void set_tree_name(const std::string &name) override ;


  void write_variable(const std::string &name, const void *p, std::type_index t_index) override;
  void write_variable(const std::string &name, const std::string *p, size_t nb_char)override ;
  void write_variable(const std::string &name, const char *p, size_t nb_char) override  ;
  void write_variable(const std::string &name, const int *p, size_t n) override  ;

  template<typename T >
  void write_variable(const std::string &name, const T *p)
  {
      register_variable(name, p);
  }

private:
  std::unordered_map<const std::string*, char*> m_mapConstStringToRootString;
  static bool s_registered;
};


class GateInputRootTreeFile: public GateRootTree, public GateInputTreeFile
{
 public:
  GateInputRootTreeFile();

  void open(const std::string& s) override ;
  virtual ~GateInputRootTreeFile();

  bool is_open() override ;
  void close() override ;
  uint64_t nb_elements() override ;


  bool data_to_read() override ;
  void read_header() override;
  void read_next_entrie() override;

  void read_entrie(const uint64_t &i) override;

  void set_tree_name(const std::string &name) override ;

  bool has_variable(const std::string &name) override;

  std::type_index get_type_of_variable(const std::string &name) override;


  using GateInputTreeFile::read_variable; //call templated version
  void read_variable(const std::string &name, std::string *p) override ;
  void read_variable(const std::string &name, char* p) override ;
  void read_variable(const std::string &name, void *p, std::type_index t_index) override ;
  void read_variable(const std::string &name, char *p, size_t nb_char) override;


private:
  void check_existence_and_kind(const std::string &name, std::type_index t_index);
  int64_t m_current_entry;
  std::unordered_map<char*, std::string*> m_mapNewStringToRefString;
  bool m_read_header_called;
  static bool s_registered;
};




