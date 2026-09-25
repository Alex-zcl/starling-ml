"""Более подробные recipes построены изменением обычных конфигов."""
from copy import deepcopy


def add(config, name, cls, params, *, creates=(), reads=(), mutates=(), updates=()):
    config['modules'][name] = {'class': cls, 'params': params}
    config['contracts'][name] = dict(creates=list(creates), reads=list(reads), mutates=list(mutates), updates=list(updates))


def special_config(task, **options):
    from .configs import get_config
    if task in {'detection', 'instance_segmentation'}:
        config = get_config('classification', **options)
        masks = task == 'instance_segmentation'
        config['modules']['dataset']['params'].update(factory='starling_ml.modules.data.general.DetectionDataset', kwargs=dict(seed=options.get('seed',42), masks=masks))
        config['modules']['model']['params'].update(factory='starling_ml.modules.models.architectures.QueryDetector', kwargs=dict(masks=masks))
        config['modules']['batch']['params']['collate_fn'] = 'starling_ml.modules.data.general.detection_collate'
        config['modules']['loss'] = {'class':'starling_ml.modules.detection.DetectionLoss','params':dict(prediction='model.output',target='batch.target')}
        return config
    if task == 'weighted_segmentation':
        config = get_config('segmentation', **options)
        add(config,'statistics','starling_ml.modules.data.statistics.DatasetStats',dict(dataset='$ctx:data.dataset', class_names=['background','A','B'], mode='multiclass'),reads=['data.dataset'],creates=['data.statistics'])
        add(config,'pixel_cost','starling_ml.modules.control.weights.DatasetWeights',dict(statistics='$ctx:data.statistics',level='element',balance='class',output='weights.label',gamma=.5),reads=['data.statistics'],creates=['weights.label'])
        add(config,'case_weights','starling_ml.modules.control.weights.DatasetWeights',dict(statistics='$ctx:data.statistics',level='case',balance='binary',positive_output='weights.present',negative_output='weights.empty',missing='neutral'),reads=['data.statistics'],creates=['weights.present','weights.empty'])
        config['modules']['loss']['class']='starling_ml.modules.losses.basic.MulticlassFocalLoss'
        config['modules']['loss']['params']['label_cost']='weights.label'
        config['contracts']['loss']['reads'].append('weights.label')
        config['modules']['dice']['params'].update(present_weight='weights.present',empty_weight='weights.empty',class_weight=[1.,1.,2.],empty_cost=.2,include_classes=[1,2])
        config['contracts']['dice']['reads'] += ['weights.present','weights.empty']
        config['metadata']=dict(task_mode='multiclass',class_names=['background','A','B'],background_class_id=0,split='train')
        return config
    if task == 'segmentation_validation':
        config = get_config('segmentation', **options)
        config['modules']['phase']['params'].update(validate_every=2,validate_at_start=True,validate_at_end=True)
        train_metric_keys=[f'metrics.train.{key}{suffix}' for key in ('precision','recall','f1','dice','iou','accuracy') for suffix in ('','_mean')]
        add(config,'train_metrics','starling_ml.modules.metrics.segmentation.SegmentationMetrics',dict(num_classes=3,prediction='model.output',target='batch.target',mode='multiclass',prefix='metrics.train',phase='train',reset_on='phase_started',finalize_on='phase_ended'),reads=['model.output','batch.target'],creates=train_metric_keys)
        train_body=config['pipeline'][-1]['while']['body']
        train_body.insert(next(i for i,step in enumerate(train_body) if step.get('call') == 'optimizer'),dict(call='train_metrics'))
        add(config,'validation_batch','starling_ml.modules.data.general.BatchSource',dict(dataset='$ctx:data.dataset',outputs={'validation.input':'input','validation.target':'target'},batch_size=4,cycle=False,shuffle=False,reset_on=['validation_start']),reads=['data.dataset'],creates=['validation.input','validation.target'])
        for key in ('input','target'):
            add(config,'validation_move_'+key,'starling_ml.modules.processing.runtime.MoveToDevice',dict(input='validation.'+key,device='$const:device'),reads=['validation.'+key],updates=['validation.'+key])
        add(config,'validation_forward','starling_ml.modules.forward.Forward',dict(model='$ctx:model.instance',input='validation.input',output='validation.output',training=False),reads=['model.instance','validation.input'],mutates=['model.instance'],creates=['validation.output'])
        # Validation objective намеренно отдельный: пользователь может заменить
        # его независимо от train CE+Dice mixer.
        add(config,'validation_loss','starling_ml.modules.losses.overlap.DiceLoss',dict(prediction='validation.output',target='validation.target',output='validation.loss',mode='multiclass',empty_target='false_positive'),reads=['validation.output','validation.target'],creates=['validation.loss'])
        metric_keys=[f'metrics.validation.{key}{suffix}' for key in ('precision','recall','f1','dice','iou','accuracy') for suffix in ('','_mean')]
        add(config,'metrics','starling_ml.modules.metrics.segmentation.SegmentationMetrics',dict(num_classes=3,prediction='validation.output',target='validation.target',mode='multiclass'),reads=['validation.output','validation.target'],creates=metric_keys)
        validation_body = [dict(call=n) for n in ('validation_move_input','validation_move_target','validation_forward','validation_loss','metrics')]
        validation_body.append({'signal': {'name': 'phase_batch_completed', 'payload': {'phase': 'validation'}}})
        validate={'when':dict(condition='$ctx:phase.due',body=[{'iterate':dict(source='validation_batch',body=validation_body)}, {'signal': {'name': 'phase_completed', 'payload': {'phase': 'validation'}}}])}
        config['pipeline'].insert(1,deepcopy(validate))
        config['pipeline'][-1]['while']['body'].append(validate)
        return config
    if task == 'gan':
        config = get_config('regression', **options)
        config['modules']['dataset']['params']['kwargs']['task']='gan'
        config['modules']['model']['params']['kwargs']['out_features']=4
        config['modules']['optimizer']['params']['loss']='loss.generator'
        config['contracts']['optimizer']['reads']=['model.instance','loss.generator']
        config['modules'].pop('loss');config['contracts'].pop('loss')
        # Discriminator получает отдельный optimizer; backward генератора проходит через замороженный D.
        add(config,'discriminator','starling_ml.modules.processing.runtime.ObjectFactory',dict(factory='starling_ml.modules.models.architectures.MLP',kwargs=dict(in_features=4,out_features=1),output='discriminator.raw'),creates=['discriminator.raw'])
        add(config,'discriminator_device','starling_ml.modules.processing.runtime.PrepareModel',dict(model='$ctx:discriminator.raw',device='$const:device',output='discriminator.instance'),reads=['discriminator.raw'],mutates=['discriminator.raw'],creates=['discriminator.instance'])
        add(config,'detach','starling_ml.modules.processing.runtime.Detach',dict(input='model.output',output='model.detached'),reads=['model.output'],creates=['model.detached'])
        for name,source in [('real','batch.input'),('fake','model.detached'),('generator','model.output')]:
            add(config,'d_'+name,'starling_ml.modules.forward.Forward',dict(model='$ctx:discriminator.instance',input=source,output='d.'+name,training=True),reads=['discriminator.instance',source],mutates=['discriminator.instance'],creates=['d.'+name])
            add(config,'loss_'+name,'starling_ml.modules.tasks.AdversarialLoss',dict(prediction='d.'+name,real=name!='fake',output='loss.'+name),reads=['d.'+name],creates=['loss.'+name])
        add(config,'mix_d','starling_ml.modules.losses.combine.LossMixer',dict(terms=[dict(loss='loss.real'),dict(loss='loss.fake')],output='loss.discriminator'),reads=['loss.real','loss.fake'],creates=['loss.discriminator'])
        add(config,'optimizer_d','starling_ml.modules.optimization.OptimizationManager',dict(model='$ctx:discriminator.instance',loss='loss.discriminator',prefix='optim.discriminator'),reads=['discriminator.instance','loss.discriminator'],mutates=['discriminator.instance'],creates=['optim.discriminator.'+k for k in ('optimizer','did_step','step','micro_step','grad_norm','skipped')])
        for name,enabled in [('freeze_d',False),('unfreeze_d',True)]:
            add(config,name,'starling_ml.modules.processing.runtime.ModelMode',dict(model='$ctx:discriminator.instance',requires_grad=enabled),reads=['discriminator.instance'],mutates=['discriminator.instance'])
        # Публичный train.loss остаётся scalar для общего мониторинга.
        add(config,'report_loss','starling_ml.modules.losses.combine.LossMixer',dict(terms=[dict(loss='loss.generator')]),reads=['loss.generator'],creates=['train.loss'])
        names=['batch','move_input','move_target','unfreeze_d','forward','detach','d_real','d_fake','loss_real','loss_fake','mix_d','optimizer_d','freeze_d','forward','d_generator','loss_generator','optimizer','report_loss']
        config['pipeline'][-1]['while']['body']=[dict(call=n) for n in names]+[dict(when=dict(condition='$ctx:optim.main.did_step',body=[dict(signal='step_completed')]))]
        return config
    raise ValueError(f'Unknown special recipe: {task}')
