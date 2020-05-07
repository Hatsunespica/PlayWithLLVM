#include "parser.h"

std::map<char,int> binopPrecedence;
int curTok;

int getNextToken(){return curTok=gettok();}

int gettok(){
  static int lastChar=' ';

  while(isspace(lastChar))lastChar=getchar();

  if(isalpha(lastChar)){
    identifierStr=lastChar;
    while(isalnum(lastChar=getchar())){
      identifierStr+=lastChar;
    }

    if(identifierStr=="def")
      return tok_def;
    else if(identifierStr=="extern")
      return tok_extern;
    return tok_identifier;
  }else if(isdigit(lastChar)||lastChar=='.'){
    std::string numStr;
    do{
      numStr+=lastChar;
      lastChar=getchar();
    }while(isdigit(lastChar)||lastChar=='.');

    numVal=strtod(numStr.c_str(),nullptr);
    return tok_number;
  }else if(lastChar=='#'){
    do{
      lastChar=getchar();
    }while(lastChar!=EOF&&lastChar!='\n'&&lastChar!='\r');
    if(lastChar!=EOF)
      return gettok();
  }

  if(lastChar==EOF)
    return tok_eof;

  int thisChar=lastChar;
  lastChar=getchar();
  return thisChar;
}

int getTokPrecedence(){
  if(!isascii(curTok))
    return -1;

  int tokPrec=binopPrecedence[curTok];
  if(tokPrec<=0)
    return -1;
  return tokPrec;
}

std::unique_ptr<ExprAST> parseNumberExpr(){
  auto result=std::make_unique<NumberExprAST>(numVal);
  getNextToken();
  return std::move(result);
}

std::unique_ptr<ExprAST> parseParenExpr(){
  getNextToken();
  auto v=parseExpression();
  if(!v)
    return nullptr;

  if(curTok!=')')
    return logError("Expected ')'");
  getNextToken();
  return v;
}

std::unique_ptr<ExprAST> parseIdentifierExpr(){
  std::string idName=identifierStr;
  getNextToken();

  if(curTok!='(')
    return std::make_unique<VariableExprAST>(idName);

  getNextToken();
  std::vector<std::unique_ptr<ExprAST>> args;
  if(curTok!=')'){
    while(true){
      if(auto arg=parseExpression()){
        args.push_back(std::move(arg));
      }else{
        return nullptr;
      }

      if(curTok==')')
        break;
      if(curTok!=',')
        return logError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  getNextToken();
  return std::make_unique<CallExprAST>(idName,std::move(args));
}

std::unique_ptr<ExprAST> parsePrimary(){
  switch(curTok){
  default:
    return logError("unknown token when expecting an expression");
  case tok_identifier:
    return parseIdentifierExpr();
  case tok_number:
    return parseNumberExpr();
  case '(':
    return parseParenExpr();
  }
}

std::unique_ptr<ExprAST> parseBinopRHS(const int exprPrec,
                                              std::unique_ptr<ExprAST> lhs){
  while(true){
    int tokPrec=getTokPrecedence();
    if(tokPrec<exprPrec)
      return lhs;

    int binOp=curTok;
    getNextToken();

    auto rhs=parsePrimary();
    if(!rhs)
      return nullptr;

    int nextPrec=getTokPrecedence();
    if(tokPrec<nextPrec){
      rhs=parseBinopRHS(tokPrec+1,std::move(rhs));
      if(!rhs)
        return nullptr;
    }
    lhs=std::make_unique<BinaryExprAST>(binOp,std::move(lhs),
                                        std::move(rhs));
  }
}

std::unique_ptr<ExprAST> parseExpression(){
  auto lhs=parsePrimary();
  if(!lhs)
    return nullptr;
  return parseBinopRHS(0,std::move(lhs));
}

std::unique_ptr<PrototypeAST> parsePrototype(){
  if(curTok!=tok_identifier)
    return logErrorP("Expected function name in prototype");

  std::string fName=identifierStr;
  getNextToken();
  if(curTok!='(')
    return logErrorP("Expected '(' in prototype");

  std::vector<std::string> argNames;
  while(getNextToken()==tok_identifier)
    argNames.push_back(identifierStr);
  if(curTok!=')')
    return logErrorP("Expected ')' in prototype");

  getNextToken();
  return std::make_unique<PrototypeAST>(fName,std::move(argNames));
}

std::unique_ptr<FunctionAST> parseDefinition(){
  getNextToken();
  auto proto=parsePrototype();
  if(!proto)
    return nullptr;

  if(auto E=parseExpression())
    return std::make_unique<FunctionAST>(std::move(proto),std::move(E));
  return nullptr;
}

std::unique_ptr<FunctionAST> parseTopLevelExpr(){
  if(auto E=parseExpression()){
    auto proto=std::make_unique<PrototypeAST>("__anon_expr",
                                              std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(proto),std::move(E));
  }
  return nullptr;
}

std::unique_ptr<PrototypeAST> parseExtern(){
  getNextToken();
  return parsePrototype();
}

