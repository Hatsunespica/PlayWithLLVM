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
    else if(identifierStr=="if")
      return tok_if;
    else if(identifierStr=="else")
      return tok_else;
    else if(identifierStr=="then")
      return tok_then;
    else if(identifierStr=="in")
      return tok_in;
    else if(identifierStr=="for")
      return tok_for;
    else if(identifierStr=="binary")
        return tok_binary;
    else if(identifierStr=="unary")
        return tok_unary;
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

/**
* numberexpr ::= number
**/

std::unique_ptr<ExprAST> parseNumberExpr(){
  auto result=std::make_unique<NumberExprAST>(numVal);
  getNextToken();
  return std::move(result);
}

/**
* parenexpr ::= '(' expression ')'
**/
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

/**
* identiferExper ::=
    identifier
    identifier '(' expression* ')'
*/
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

/**
* primary::=
    identifier
    numberexpr
    parenexpr
    ifexpr
    forexpr
*/
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
  case tok_if:
    return parseIfExpr();
  case tok_for:
    return parseForExpr();
  }
}

/**
* binoRHS
    ('+' unary)*
*/
std::unique_ptr<ExprAST> parseBinopRHS(const int exprPrec,
                                              std::unique_ptr<ExprAST> lhs){
  while(true){
    int tokPrec=getTokPrecedence();
    if(tokPrec<exprPrec)
      return lhs;

    int binOp=curTok;
    getNextToken();

    auto rhs=parseUnary();
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

/**
* expression
    ::= unary binoprhs
*/
std::unique_ptr<ExprAST> parseExpression(){
  auto lhs=parseUnary();
  if(!lhs)
    return nullptr;
  return parseBinopRHS(0,std::move(lhs));
}

/**
* prototype
    ::= id '(' id* ')'
    ::= binary LETTER number? (id,id)
    ::= unary LETTER (id)
*/
std::unique_ptr<PrototypeAST> parsePrototype(){
    std::string fName;
    unsigned kind=0,binaryPrecedence=30;
    switch(curTok){
        default:
            return logErrorP("Expected function name in prototype");
        case tok_identifier:
            fName=identifierStr;
            getNextToken();
            break;
        case tok_unary:
            getNextToken();
            if(!isascii(curTok))
                return logErrorP("Expected unary operator");
            fName="unary";
            fName+=(char)curTok;
            kind=1;
            getNextToken();
            break;
        case tok_binary:
            getNextToken();
            if(!isascii(curTok))
                return logErrorP("Expected binary operator");
            fName="binary";
            fName+=(char)curTok;
            kind=2;
            getNextToken();
            if(curTok==tok_number){
                if(numVal<1||numVal>100)
                    return logErrorP("Invalid precedence: must be 1 ..100");
                binaryPrecedence=(unsigned)numVal;
                getNextToken();
            }
            break;
    }

    if(curTok!='(')
        return logErrorP("Expected '(' in prototype");

    std::vector<std::string> argNames;
    while(getNextToken()==tok_identifier)
        argNames.push_back(identifierStr);
    if(curTok!=')')
        return logErrorP("Expected ')' in prototype");
    getNextToken();
    if(kind&&argNames.size()!=kind)
        return logErrorP("Invalid number of operands for operator");
    return std::make_unique<PrototypeAST>(fName,std::move(argNames),kind!=0,binaryPrecedence);
}

/**
* definition ::= 'def' prototype expression
*/
std::unique_ptr<FunctionAST> parseDefinition(){
  getNextToken();
  auto proto=parsePrototype();
  if(!proto)
    return nullptr;

  if(auto E=parseExpression())
    return std::make_unique<FunctionAST>(std::move(proto),std::move(E));
  return nullptr;
}

/**
* toplevelexpre ::= expression
*/
std::unique_ptr<FunctionAST> parseTopLevelExpr(){
  if(auto E=parseExpression()){
    auto proto=std::make_unique<PrototypeAST>("__anon_expr",
                                              std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(proto),std::move(E));
  }
  return nullptr;
}

/**
* external ::= 'extern' prototype
*/
std::unique_ptr<PrototypeAST> parseExtern(){
  getNextToken();
  return parsePrototype();
}

/**
* ifexpr ::=
    'if' expression 'then' expression 'else' expression
*/
std::unique_ptr<ExprAST> parseIfExpr(){
    getNextToken();

    auto cond=parseExpression();
    if(!cond)
        return nullptr;

    if(curTok!=tok_then)
        return logError("expected then");
    getNextToken();

    auto then=parseExpression();
    if(!then)
        return nullptr;

    if(curTok!=tok_else)
        return logError("expected else");

    getNextToken();
    auto els=parseExpression();
    if(!els)
        return nullptr;

    return std::make_unique<IfExprAST>(std::move(cond),std::move(then),std::move(els));
}

/**
* forexpr ::= 'for' identifier '=' expression ',' expression  (',' expr)? 'in' expression
*/
std::unique_ptr<ExprAST> parseForExpr(){
    getNextToken();
    if(curTok!=tok_identifier)
      return logError("expected identifier after for");

    std::string idName=identifierStr;
    getNextToken();

    if(curTok!='=')
      return logError("expected '=' after for");
    getNextToken();

    auto start=parseExpression();
    if(!start)
        return nullptr;
    if(curTok!=',')
      return logError("expected ',' after for start value");
    getNextToken();

    auto end=parseExpression();
    if(!end)
        return nullptr;

     std::unique_ptr<ExprAST> step;
     if(curTok==','){
        getNextToken();
        step=parseExpression();
        if(!step)
            return nullptr;
     }

     if(curTok!=tok_in)
        return logError("expected 'in' after for");
     getNextToken();

      auto body=parseExpression();
      if(!body)
        return nullptr;

      return std::make_unique<ForExprAST>(idName,std::move(start),std::move(end),std::move(step),std::move(body));
}

/**
* unary::=
    primary
    '!' unary
*/
std::unique_ptr<ExprAST> parseUnary(){
    if(!isascii(curTok)||curTok=='('||curTok==',')
        return parsePrimary();
    int opc=curTok;
    getNextToken();
    if(auto operand=parseUnary())
        return std::make_unique<UnaryExprAST>(opc,std::move(operand));
    return nullptr;
}
