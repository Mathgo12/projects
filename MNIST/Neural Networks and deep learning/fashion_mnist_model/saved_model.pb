??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
my_model_10/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namemy_model_10/dense_32/kernel
?
/my_model_10/dense_32/kernel/Read/ReadVariableOpReadVariableOpmy_model_10/dense_32/kernel* 
_output_shapes
:
??*
dtype0
?
my_model_10/dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namemy_model_10/dense_32/bias
?
-my_model_10/dense_32/bias/Read/ReadVariableOpReadVariableOpmy_model_10/dense_32/bias*
_output_shapes	
:?*
dtype0
?
my_model_10/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*,
shared_namemy_model_10/dense_33/kernel
?
/my_model_10/dense_33/kernel/Read/ReadVariableOpReadVariableOpmy_model_10/dense_33/kernel*
_output_shapes
:	?
*
dtype0
?
my_model_10/dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namemy_model_10/dense_33/bias
?
-my_model_10/dense_33/bias/Read/ReadVariableOpReadVariableOpmy_model_10/dense_33/bias*
_output_shapes
:
*
dtype0
?
my_model_10/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*,
shared_namemy_model_10/dense_34/kernel
?
/my_model_10/dense_34/kernel/Read/ReadVariableOpReadVariableOpmy_model_10/dense_34/kernel*
_output_shapes

:

*
dtype0
?
my_model_10/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namemy_model_10/dense_34/bias
?
-my_model_10/dense_34/bias/Read/ReadVariableOpReadVariableOpmy_model_10/dense_34/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
hidden1
hidden2
mainout
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
h


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
6
iter
	decay
learning_rate
momentum
 
*

0
1
2
3
4
5
*

0
1
2
3
4
5
?
 non_trainable_variables

!layers
"layer_regularization_losses
regularization_losses
#layer_metrics
$metrics
trainable_variables
	variables
 
ZX
VARIABLE_VALUEmy_model_10/dense_32/kernel)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEmy_model_10/dense_32/bias'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUE
 


0
1


0
1
?
%non_trainable_variables

&layers
'layer_regularization_losses
regularization_losses
(layer_metrics
)metrics
trainable_variables
	variables
ZX
VARIABLE_VALUEmy_model_10/dense_33/kernel)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEmy_model_10/dense_33/bias'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
*non_trainable_variables

+layers
,layer_regularization_losses
regularization_losses
-layer_metrics
.metrics
trainable_variables
	variables
ZX
VARIABLE_VALUEmy_model_10/dense_34/kernel)mainout/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEmy_model_10/dense_34/bias'mainout/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
/non_trainable_variables

0layers
1layer_regularization_losses
regularization_losses
2layer_metrics
3metrics
trainable_variables
	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 
 

40
51
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	6total
	7count
8	variables
9	keras_api
D
	:total
	;count
<
_fn_kwargs
=	variables
>	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

60
71

8	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

=	variables
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_10/dense_32/kernelmy_model_10/dense_32/biasmy_model_10/dense_33/kernelmy_model_10/dense_33/biasmy_model_10/dense_34/kernelmy_model_10/dense_34/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_680856
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/my_model_10/dense_32/kernel/Read/ReadVariableOp-my_model_10/dense_32/bias/Read/ReadVariableOp/my_model_10/dense_33/kernel/Read/ReadVariableOp-my_model_10/dense_33/bias/Read/ReadVariableOp/my_model_10/dense_34/kernel/Read/ReadVariableOp-my_model_10/dense_34/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_681069
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_10/dense_32/kernelmy_model_10/dense_32/biasmy_model_10/dense_33/kernelmy_model_10/dense_33/biasmy_model_10/dense_34/kernelmy_model_10/dense_34/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_681121??
?
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680754
input_1
dense_32_680738
dense_32_680740
dense_33_680743
dense_33_680745
dense_34_680748
dense_34_680750
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCallo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeinput_1flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0dense_32_680738dense_32_680740*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_6806622"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_680743dense_33_680745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_6806892"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_680748dense_34_680750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_6807162"
 dense_34/StatefulPartitionedCall?
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
~
)__inference_dense_32_layer_call_fn_680964

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_6806622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_my_model_10_layer_call_fn_680927

input_
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_10_layer_call_and_return_conditional_losses_6807782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinput_
? 
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680910

input_+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity??dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeinput_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMulflatten/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAddt
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_32/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_33/BiasAdds
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_33/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_34/BiasAdd|
dense_34/SoftmaxSoftmaxdense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_34/Softmax?
IdentityIdentitydense_34/Softmax:softmax:0 ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinput_
?	
?
D__inference_dense_34_layer_call_and_return_conditional_losses_680716

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_dense_34_layer_call_and_return_conditional_losses_680995

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
,__inference_my_model_10_layer_call_fn_680831
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_10_layer_call_and_return_conditional_losses_6808162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680733
input_1
dense_32_680673
dense_32_680675
dense_33_680700
dense_33_680702
dense_34_680727
dense_34_680729
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCallo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeinput_1flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0dense_32_680673dense_32_680675*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_6806622"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_680700dense_33_680702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_6806892"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_680727dense_34_680729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_6807162"
 dense_34/StatefulPartitionedCall?
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
,__inference_my_model_10_layer_call_fn_680944

input_
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_10_layer_call_and_return_conditional_losses_6808162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinput_
?
?
$__inference_signature_wrapper_680856
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_6806452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680816

input_
dense_32_680800
dense_32_680802
dense_33_680805
dense_33_680807
dense_34_680810
dense_34_680812
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCallo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeinput_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0dense_32_680800dense_32_680802*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_6806622"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_680805dense_33_680807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_6806892"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_680810dense_34_680812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_6807162"
 dense_34/StatefulPartitionedCall?
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinput_
?
?
,__inference_my_model_10_layer_call_fn_680793
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_my_model_10_layer_call_and_return_conditional_losses_6807782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
D__inference_dense_32_layer_call_and_return_conditional_losses_680662

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_33_layer_call_and_return_conditional_losses_680975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_33_layer_call_fn_680984

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_6806892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_32_layer_call_and_return_conditional_losses_680955

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
"__inference__traced_restore_681121
file_prefix0
,assignvariableop_my_model_10_dense_32_kernel0
,assignvariableop_1_my_model_10_dense_32_bias2
.assignvariableop_2_my_model_10_dense_33_kernel0
,assignvariableop_3_my_model_10_dense_33_bias2
.assignvariableop_4_my_model_10_dense_34_kernel0
,assignvariableop_5_my_model_10_dense_34_bias
assignvariableop_6_sgd_iter 
assignvariableop_7_sgd_decay(
$assignvariableop_8_sgd_learning_rate#
assignvariableop_9_sgd_momentum
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUEB)mainout/kernel/.ATTRIBUTES/VARIABLE_VALUEB'mainout/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_my_model_10_dense_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_my_model_10_dense_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_my_model_10_dense_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_my_model_10_dense_33_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_my_model_10_dense_34_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_my_model_10_dense_34_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?(
?
!__inference__wrapped_model_680645
input_17
3my_model_10_dense_32_matmul_readvariableop_resource8
4my_model_10_dense_32_biasadd_readvariableop_resource7
3my_model_10_dense_33_matmul_readvariableop_resource8
4my_model_10_dense_33_biasadd_readvariableop_resource7
3my_model_10_dense_34_matmul_readvariableop_resource8
4my_model_10_dense_34_biasadd_readvariableop_resource
identity??+my_model_10/dense_32/BiasAdd/ReadVariableOp?*my_model_10/dense_32/MatMul/ReadVariableOp?+my_model_10/dense_33/BiasAdd/ReadVariableOp?*my_model_10/dense_33/MatMul/ReadVariableOp?+my_model_10/dense_34/BiasAdd/ReadVariableOp?*my_model_10/dense_34/MatMul/ReadVariableOp?
my_model_10/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
my_model_10/flatten/Const?
my_model_10/flatten/ReshapeReshapeinput_1"my_model_10/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
my_model_10/flatten/Reshape?
*my_model_10/dense_32/MatMul/ReadVariableOpReadVariableOp3my_model_10_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*my_model_10/dense_32/MatMul/ReadVariableOp?
my_model_10/dense_32/MatMulMatMul$my_model_10/flatten/Reshape:output:02my_model_10/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
my_model_10/dense_32/MatMul?
+my_model_10/dense_32/BiasAdd/ReadVariableOpReadVariableOp4my_model_10_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+my_model_10/dense_32/BiasAdd/ReadVariableOp?
my_model_10/dense_32/BiasAddBiasAdd%my_model_10/dense_32/MatMul:product:03my_model_10/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
my_model_10/dense_32/BiasAdd?
my_model_10/dense_32/ReluRelu%my_model_10/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
my_model_10/dense_32/Relu?
*my_model_10/dense_33/MatMul/ReadVariableOpReadVariableOp3my_model_10_dense_33_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*my_model_10/dense_33/MatMul/ReadVariableOp?
my_model_10/dense_33/MatMulMatMul'my_model_10/dense_32/Relu:activations:02my_model_10/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
my_model_10/dense_33/MatMul?
+my_model_10/dense_33/BiasAdd/ReadVariableOpReadVariableOp4my_model_10_dense_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+my_model_10/dense_33/BiasAdd/ReadVariableOp?
my_model_10/dense_33/BiasAddBiasAdd%my_model_10/dense_33/MatMul:product:03my_model_10/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
my_model_10/dense_33/BiasAdd?
my_model_10/dense_33/ReluRelu%my_model_10/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
my_model_10/dense_33/Relu?
*my_model_10/dense_34/MatMul/ReadVariableOpReadVariableOp3my_model_10_dense_34_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02,
*my_model_10/dense_34/MatMul/ReadVariableOp?
my_model_10/dense_34/MatMulMatMul'my_model_10/dense_33/Relu:activations:02my_model_10/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
my_model_10/dense_34/MatMul?
+my_model_10/dense_34/BiasAdd/ReadVariableOpReadVariableOp4my_model_10_dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+my_model_10/dense_34/BiasAdd/ReadVariableOp?
my_model_10/dense_34/BiasAddBiasAdd%my_model_10/dense_34/MatMul:product:03my_model_10/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
my_model_10/dense_34/BiasAdd?
my_model_10/dense_34/SoftmaxSoftmax%my_model_10/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
my_model_10/dense_34/Softmax?
IdentityIdentity&my_model_10/dense_34/Softmax:softmax:0,^my_model_10/dense_32/BiasAdd/ReadVariableOp+^my_model_10/dense_32/MatMul/ReadVariableOp,^my_model_10/dense_33/BiasAdd/ReadVariableOp+^my_model_10/dense_33/MatMul/ReadVariableOp,^my_model_10/dense_34/BiasAdd/ReadVariableOp+^my_model_10/dense_34/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2Z
+my_model_10/dense_32/BiasAdd/ReadVariableOp+my_model_10/dense_32/BiasAdd/ReadVariableOp2X
*my_model_10/dense_32/MatMul/ReadVariableOp*my_model_10/dense_32/MatMul/ReadVariableOp2Z
+my_model_10/dense_33/BiasAdd/ReadVariableOp+my_model_10/dense_33/BiasAdd/ReadVariableOp2X
*my_model_10/dense_33/MatMul/ReadVariableOp*my_model_10/dense_33/MatMul/ReadVariableOp2Z
+my_model_10/dense_34/BiasAdd/ReadVariableOp+my_model_10/dense_34/BiasAdd/ReadVariableOp2X
*my_model_10/dense_34/MatMul/ReadVariableOp*my_model_10/dense_34/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
~
)__inference_dense_34_layer_call_fn_681004

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_6807162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?&
?
__inference__traced_save_681069
file_prefix:
6savev2_my_model_10_dense_32_kernel_read_readvariableop8
4savev2_my_model_10_dense_32_bias_read_readvariableop:
6savev2_my_model_10_dense_33_kernel_read_readvariableop8
4savev2_my_model_10_dense_33_bias_read_readvariableop:
6savev2_my_model_10_dense_34_kernel_read_readvariableop8
4savev2_my_model_10_dense_34_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUEB)mainout/kernel/.ATTRIBUTES/VARIABLE_VALUEB'mainout/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_my_model_10_dense_32_kernel_read_readvariableop4savev2_my_model_10_dense_32_bias_read_readvariableop6savev2_my_model_10_dense_33_kernel_read_readvariableop4savev2_my_model_10_dense_33_bias_read_readvariableop6savev2_my_model_10_dense_34_kernel_read_readvariableop4savev2_my_model_10_dense_34_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*[
_input_shapesJ
H: :
??:?:	?
:
:

:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680778

input_
dense_32_680762
dense_32_680764
dense_33_680767
dense_33_680769
dense_34_680772
dense_34_680774
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCallo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeinput_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0dense_32_680762dense_32_680764*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_6806622"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_680767dense_33_680769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_6806892"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_680772dense_34_680774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_6807162"
 dense_34/StatefulPartitionedCall?
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinput_
?	
?
D__inference_dense_33_layer_call_and_return_conditional_losses_680689

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680883

input_+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity??dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeinput_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMulflatten/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAddt
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_32/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_33/BiasAdds
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_33/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_34/BiasAdd|
dense_34/SoftmaxSoftmaxdense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_34/Softmax?
IdentityIdentitydense_34/Softmax:softmax:0 ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinput_"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:?g
?
hidden1
hidden2
mainout
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
?__call__
*@&call_and_return_all_conditional_losses
A_default_save_signature"?
_tf_keras_model?{"class_name": "MyModel", "name": "my_model_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
B__call__
*C&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
I
iter
	decay
learning_rate
momentum"
	optimizer
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
?
 non_trainable_variables

!layers
"layer_regularization_losses
regularization_losses
#layer_metrics
$metrics
trainable_variables
	variables
?__call__
A_default_save_signature
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
,
Hserving_default"
signature_map
/:-
??2my_model_10/dense_32/kernel
(:&?2my_model_10/dense_32/bias
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
%non_trainable_variables

&layers
'layer_regularization_losses
regularization_losses
(layer_metrics
)metrics
trainable_variables
	variables
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
.:,	?
2my_model_10/dense_33/kernel
':%
2my_model_10/dense_33/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
*non_trainable_variables

+layers
,layer_regularization_losses
regularization_losses
-layer_metrics
.metrics
trainable_variables
	variables
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
-:+

2my_model_10/dense_34/kernel
':%
2my_model_10/dense_34/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
/non_trainable_variables

0layers
1layer_regularization_losses
regularization_losses
2layer_metrics
3metrics
trainable_variables
	variables
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	6total
	7count
8	variables
9	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	:total
	;count
<
_fn_kwargs
=	variables
>	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
60
71"
trackable_list_wrapper
-
8	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
-
=	variables"
_generic_user_object
?2?
,__inference_my_model_10_layer_call_fn_680927
,__inference_my_model_10_layer_call_fn_680944
,__inference_my_model_10_layer_call_fn_680831
,__inference_my_model_10_layer_call_fn_680793?
???
FullArgSpec1
args)?&
jself
jinput_

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680883
G__inference_my_model_10_layer_call_and_return_conditional_losses_680910
G__inference_my_model_10_layer_call_and_return_conditional_losses_680754
G__inference_my_model_10_layer_call_and_return_conditional_losses_680733?
???
FullArgSpec1
args)?&
jself
jinput_

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_680645?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_1?????????
?2?
)__inference_dense_32_layer_call_fn_680964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_32_layer_call_and_return_conditional_losses_680955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_33_layer_call_fn_680984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_33_layer_call_and_return_conditional_losses_680975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_34_layer_call_fn_681004?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_34_layer_call_and_return_conditional_losses_680995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_680856input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_680645s
4?1
*?'
%?"
input_1?????????
? "3?0
.
output_1"?
output_1?????????
?
D__inference_dense_32_layer_call_and_return_conditional_losses_680955^
0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_32_layer_call_fn_680964Q
0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_33_layer_call_and_return_conditional_losses_680975]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? }
)__inference_dense_33_layer_call_fn_680984P0?-
&?#
!?
inputs??????????
? "??????????
?
D__inference_dense_34_layer_call_and_return_conditional_losses_680995\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????

? |
)__inference_dense_34_layer_call_fn_681004O/?,
%?"
 ?
inputs?????????

? "??????????
?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680733m
<?9
2?/
%?"
input_1?????????
p

 
? "%?"
?
0?????????

? ?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680754m
<?9
2?/
%?"
input_1?????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680883l
;?8
1?.
$?!
input_?????????
p

 
? "%?"
?
0?????????

? ?
G__inference_my_model_10_layer_call_and_return_conditional_losses_680910l
;?8
1?.
$?!
input_?????????
p 

 
? "%?"
?
0?????????

? ?
,__inference_my_model_10_layer_call_fn_680793`
<?9
2?/
%?"
input_1?????????
p

 
? "??????????
?
,__inference_my_model_10_layer_call_fn_680831`
<?9
2?/
%?"
input_1?????????
p 

 
? "??????????
?
,__inference_my_model_10_layer_call_fn_680927_
;?8
1?.
$?!
input_?????????
p

 
? "??????????
?
,__inference_my_model_10_layer_call_fn_680944_
;?8
1?.
$?!
input_?????????
p 

 
? "??????????
?
$__inference_signature_wrapper_680856~
??<
? 
5?2
0
input_1%?"
input_1?????????"3?0
.
output_1"?
output_1?????????
