"""Математические инварианты новой версии: проверяем значения и gradients."""
import unittest
import torch
import torch.nn.functional as F
from starling_ml.ops.pointwise import binary_objective, categorical_loss_objective
from starling_ml.ops.losses import overlap_objective, generalized_dice_objective
from starling_ml.ops.reduction import pair_objective, pointwise_objective
from starling_ml.ops.statistics import DatasetStatistics, frequency_factors, binary_frequency_factors
from starling_ml.ops.weights import presence_case_weights
from starling_ml.ops.metrics import multilabel_stats, multiclass_stats


class MathTests(unittest.TestCase):
    def test_binary_cost_is_not_in_denominator(self):
        x=torch.zeros(2,1);y=torch.tensor([[1.],[0.]])
        result=binary_objective(x,y,positive_cost=3,negative_cost=1).value()
        self.assertAlmostEqual(result.item(),2*torch.log(torch.tensor(2.)).item())

    def test_binary_soft_target_focal_components(self):
        x=torch.tensor([[.7]],dtype=torch.float64);y=torch.tensor([[.3]])
        result=binary_objective(x,y,gamma=2,positive_cost=3,negative_cost=2).value()
        expected=3*y*F.softplus(-x)*torch.sigmoid(-x)**2+2*(1-y)*F.softplus(x)*torch.sigmoid(x)**2
        torch.testing.assert_close(result,expected.squeeze())

    def test_focal_probability_is_unweighted(self):
        x=torch.tensor([[1.,-.3],[.2,.8]],dtype=torch.float64);y=torch.tensor([1,0]);cost=torch.tensor([2.,5.])
        logp=x.log_softmax(1);i=torch.arange(2)
        expected=(-cost[y]*(1-logp[i,y].exp())**2*logp[i,y]).mean()
        torch.testing.assert_close(categorical_loss_objective(x,y,gamma=2,label_cost=cost).value(),expected)

    def test_soft_targets_channel_last(self):
        x=torch.randn(2,4,3);y=x.softmax(-1)
        expected=-(y*x.log_softmax(-1)).sum(-1).mean()
        torch.testing.assert_close(categorical_loss_objective(x,y,class_dim=-1).value(),expected)

    def test_ignore_sample_does_not_dilute_ce(self):
        x=torch.zeros(2,3);y=torch.tensor([1,-100])
        torch.testing.assert_close(categorical_loss_objective(x,y).value(),torch.log(torch.tensor(3.)))

    def test_zero_fp16_weights_have_finite_backward(self):
        for dtype in (torch.float16,torch.bfloat16):
            with self.subTest(dtype=dtype):
                x=torch.zeros(2,2,64,64,dtype=dtype,requires_grad=True)
                obj=binary_objective(x,torch.zeros_like(x),element_weight=0)
                value=obj.value();value.backward()
                self.assertEqual(value.item(),0);self.assertTrue(torch.isfinite(x.grad).all())

    def test_fp16_large_sums(self):
        x=torch.zeros(1,1,300,300,dtype=torch.float16,requires_grad=True)
        loss=overlap_objective(x,torch.ones_like(x),empty_target='standard')
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())

    def test_classification_element_weights_are_preserved(self):
        loss=torch.tensor([1.,3.])
        result=pointwise_objective(loss,element_weight=torch.tensor([1.,3.])).value()
        self.assertEqual(result.item(),2.5)

    def test_class_macro_uses_active_classes(self):
        loss=torch.tensor([[.2,.8],[.2,.1]])
        valid=torch.tensor([[True,True],[True,False]])
        self.assertAlmostEqual(pair_objective(loss,valid,reduction='pair_mean').value().item(),.4,places=6)
        self.assertAlmostEqual(pair_objective(loss,valid,reduction='class_macro').value().item(),.5,places=6)

    def test_all_ignored_generalized_zero(self):
        x=torch.randn(2,3,4,4,requires_grad=True);y=torch.full((2,4,4),-100)
        loss=generalized_dice_objective(x,y,smooth=0)
        loss.backward();self.assertEqual(loss.item(),0);self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(x.grad.abs().sum(),0)

    def test_empty_importance_cancels_but_cost_remains(self):
        x=torch.zeros(2,1,2,2);y=torch.zeros_like(x)
        kwargs=dict(empty_target='false_positive')
        base=overlap_objective(x,y,**kwargs)
        weighted=overlap_objective(x,y,empty_weight=.1,**kwargs)
        costed=overlap_objective(x,y,empty_cost=.1,**kwargs)
        torch.testing.assert_close(base,weighted);torch.testing.assert_close(costed,base*.1)

    def test_present_empty_mass(self):
        x=torch.zeros(2,1,1);y=torch.tensor([[[1.]],[[0.]]])
        a=overlap_objective(x[:1],y[:1],smooth=0,empty_target='false_positive')
        b=overlap_objective(x[1:],y[1:],smooth=0,empty_target='false_positive')
        result=overlap_objective(x,y,smooth=0,empty_target='false_positive',present_weight=3,empty_weight=1)
        torch.testing.assert_close(result,(3*a+b)/4)

    def test_inner_weight_changes_false_positive(self):
        x=torch.zeros(1,1,2);y=torch.tensor([[[1.,0.]]])
        first=overlap_objective(x,y,smooth=0,positive_weight=1,negative_weight=1)
        second=overlap_objective(x,y,smooth=0,positive_weight=1,negative_weight=3)
        self.assertGreater(second.item(),first.item())

    def test_background_term_exclusion_keeps_background_pixels(self):
        x=torch.tensor([[[[2.,2.]],[[0.,0.]]]],requires_grad=True);y=torch.tensor([[[0,1]]])
        loss=overlap_objective(x,y,mode='multiclass',activation='softmax',include_classes=[1],smooth=0)
        loss.backward();self.assertGreater(x.grad[0,1,0,0].abs().item(),0)

    def test_exact_merge_value_and_gradient(self):
        torch.manual_seed(4)
        target=torch.randint(3,(5,2,2));target[1,0,0]=-100
        for reduction in ('element_mean','sample_mean'):
            x=torch.randn(5,3,2,2,dtype=torch.float64,requires_grad=True)
            whole=categorical_loss_objective(x,target,reduction=reduction).value()
            gradient=torch.autograd.grad(whole,x)[0]
            y=x.detach().clone().requires_grad_(True)
            a=categorical_loss_objective(y[:2],target[:2],reduction=reduction)
            b=categorical_loss_objective(y[2:],target[2:],reduction=reduction)
            merged=a.merge(b).value()
            torch.testing.assert_close(merged,whole);torch.testing.assert_close(torch.autograd.grad(merged,y)[0],gradient)

    def test_overlap_merge_all_reducers(self):
        torch.manual_seed(9)
        target=torch.randint(2,(5,2,3,3)).double()
        for aggregation,reduction in [('sample','pair_mean'),('sample','sample_mean'),('sample','class_macro'),('batch_stats','pair_mean')]:
            with self.subTest(aggregation=aggregation,reduction=reduction):
                x=torch.randn_like(target,requires_grad=True)
                kw=dict(aggregation=aggregation,reduction=reduction,empty_target='false_positive',return_objective=True)
                whole=overlap_objective(x,target,**kw).value();grad=torch.autograd.grad(whole,x)[0]
                y=x.detach().clone().requires_grad_(True)
                merged=overlap_objective(y[:2],target[:2],**kw).merge(overlap_objective(y[2:],target[2:],**kw)).value()
                torch.testing.assert_close(merged,whole);torch.testing.assert_close(torch.autograd.grad(merged,y)[0],grad)

    def test_counts_deduplicate_studies_and_preserve_unknown(self):
        stats=DatasetStatistics(['a','b'],sample_unit='patch')
        stats.update(torch.tensor([[1,0],[0,-100]]),'study')
        stats.update(torch.tensor([[0,0],[0,0]]),'study')
        result=stats.result()
        self.assertEqual(result['study_count'],1);self.assertEqual(result['sample_count'],2)
        self.assertEqual(result['positive_study_count'].tolist(),[1,0]);self.assertEqual(result['unknown_presence_study_count'].tolist(),[0,1])
        self.assertEqual(result['negative_element_count'].tolist(),[3,3])

    def test_counts_multiclass(self):
        stats=DatasetStatistics(['background','a'],mode='multiclass')
        stats.update(torch.tensor([0,1,-100]),'x')
        self.assertEqual(stats.result()['valid_element_count'].tolist(),[2,2])

    def test_balanced_presence_factors(self):
        pos,neg=binary_frequency_factors([20],[80],gamma=1)
        torch.testing.assert_close(pos,torch.tensor([2.5]));torch.testing.assert_close(neg,torch.tensor([.625]))

    def test_absent_counts_explicit(self):
        with self.assertRaises(ValueError):frequency_factors([0,1])
        torch.testing.assert_close(frequency_factors([0,0],missing='neutral'),torch.ones(2))
        torch.testing.assert_close(frequency_factors([0,0],missing='zero'),torch.zeros(2))

    def test_presence_ignores_unknown(self):
        y=torch.tensor([[[1,-100]], [[-100,-100]]])
        torch.testing.assert_close(presence_case_weights(y,3,1),torch.tensor([[3.],[0.]]))

    def test_metrics_ignore_unknown(self):
        logits=torch.zeros(1,1,3);target=torch.tensor([[[1.,0.,-100.]]])
        tp,fp,fn,tn=multilabel_stats(logits,target)
        self.assertEqual([v.item() for v in (tp,fp,fn,tn)],[1,1,0,0])
        stats=multiclass_stats(torch.zeros(1,2,3),torch.tensor([[0,1,-100]]),2)
        self.assertEqual(sum(v.sum().item() for v in stats),4)

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):binary_objective(torch.zeros(1,1),torch.ones(1,1),positive_cost=-1)

    def test_noninteger_multiclass_rejected(self):
        with self.assertRaises(ValueError):overlap_objective(torch.zeros(1,2,1),torch.tensor([[.2]]),mode='multiclass')


if __name__=='__main__':unittest.main()


class AdditionalMathTests(unittest.TestCase):
    def test_binary_vector_with_channel_cost(self):
        x=torch.zeros(3,requires_grad=True);y=torch.tensor([0.,1.,1.])
        value=binary_objective(x,y,positive_cost=[2.]).value();value.backward()
        self.assertEqual(x.grad.shape,x.shape);self.assertTrue(torch.isfinite(value))

    def test_valid_mask_excludes_arbitrary_void_labels(self):
        x=torch.zeros(1,2,3,requires_grad=True);y=torch.tensor([[0,-1,1]]);mask=torch.tensor([[1,0,1]])
        loss=categorical_loss_objective(x,y,valid_mask=mask).value()
        dice=overlap_objective(x,y,mode='multiclass',activation='softmax',valid_mask=mask)
        (loss+dice).backward();self.assertEqual(x.grad[:,:,1].abs().sum(),0)

    def test_pixel_mean_and_pair_mean_are_distinct(self):
        values=torch.tensor([[[1.,1.]],[[3.,3.]]]);mask=torch.tensor([[[1,1]],[[1,0]]])
        self.assertAlmostEqual(pointwise_objective(values,valid_mask=mask,reduction='element_mean').value().item(),5/3,places=6)
        self.assertEqual(pointwise_objective(values,valid_mask=mask,reduction='pair_mean').value().item(),2)

    def test_zero_cost_case_stays_in_valid_count(self):
        values=torch.ones(2,1,3);weights=torch.tensor([[[0.]],[[1.]]])
        result=pointwise_objective(values,element_weight=weights,element_normalization='valid_count').value()
        self.assertEqual(result.item(),.5)

    def test_negative_patch_is_unknown_study(self):
        stats=DatasetStatistics(['a'],sample_unit='patch')
        stats.update(torch.zeros(1,3,3),'study')
        self.assertEqual(stats.result()['negative_sample_count'].item(),1)
        self.assertEqual(stats.result()['unknown_presence_study_count'].item(),1)

    def test_gae_truncation_bootstraps_without_next_episode_trace(self):
        from starling_ml.ops.tasks import generalized_advantage
        advantage,returns=generalized_advantage(torch.tensor([1.,10.]),torch.zeros(2),torch.tensor([2.,5.]),torch.tensor([False,True]),torch.tensor([True,False]),gamma=.5,lam=1.)
        torch.testing.assert_close(advantage,torch.tensor([2.,10.]))

    def test_detection_perfect_ap(self):
        from starling_ml.ops.detection import detection_ap
        target=dict(labels=torch.tensor([0]),boxes=torch.tensor([[0.,0.,1.,1.]]))
        prediction=dict(target,scores=torch.tensor([.9]))
        self.assertEqual(detection_ap([prediction],[target],1)['map'],1.)
